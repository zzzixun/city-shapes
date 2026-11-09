"""城市裡藏了哪些圖案？用 Quick, Draw! 分類器當評審，不必自己畫範本。

做法：
  1. 範本直接取自 Quick, Draw! 的手繪圖：轉成實心剪影取外輪廓，只留「只看外輪廓分類器也認得」的圖（每類 5 張）
  2. 用 shapes.py 的街廓貼合粗搜，找出每張範本最貼合的幾個位置，合併街廓得到路線
  3. 路線畫成 28×28 手繪風格，交給分類器打分；以「分類器信心 + 貼合度 − 鋸齒」再微調
  4. 每類留下最好的路線；依信心排序就是「這個行政區最像什麼」

用法：python discover.py daan      （需先跑 fetch.py daan、train.py）
輸出：out/<名稱>/discover.png、discover.json、discover/*.gpx；out/<名稱>/recognize.json（舊範本結果的辨識）
"""
import json
import math
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon
from skimage import measure

from qd import CLASSES, DATA, ZH, load_model, outline_style, predict, render, silhouette
from shapes import ZIGZAG, City, far_apart, route_of, to_gpx

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]

N_TPL = 5  # 每類取幾張手繪圖當範本
SIZES = [800, 1200]
ANGLES = [-20, 0, 20]
STEP = 100
SEEDS = 4  # 每張範本留幾個位置
REFINE = 30  # 微調步數
NET = load_model()
K = {c: i for i, c in enumerate(CLASSES)}


def templates(c, n=N_TPL, max_complexity=40):
    """從 Quick, Draw! 挑出外輪廓就認得出來的手繪圖，轉成 [-1, 1] 的多邊形。"""
    a = np.load(f"{DATA}/{c}.npy", mmap_mode="r")
    imgs = np.array(a[-3000:]).reshape(-1, 28, 28)  # 取檔案最後面，和訓練資料大致錯開
    sil = [silhouette(x) for x in imgs]
    ok = [i for i, s in enumerate(sil) if s.sum() > 150 and ndimage.label(s)[1] == 1]
    p = predict(NET, np.stack([outline_style(imgs[i]) for i in ok]))[:, K[c]]
    out = []
    for i in np.array(ok)[np.argsort(-p)]:
        big = ndimage.zoom(sil[i].astype(float), 4, order=1)
        ring = max(measure.find_contours(np.pad(big, 2), 0.5), key=len)
        poly = Polygon(ring[:, ::-1] * [1, -1]).simplify(1.5)  # (列, 行) → (x, y 朝上)
        xy = np.array(poly.exterior.coords[:-1])
        xy = xy - (xy.min(0) + xy.max(0)) / 2
        xy = xy / np.abs(xy).max()
        # 太複雜（周長² / 面積 太大）的跑起來會一直鑽巷子，跳過
        if poly.is_valid and poly.length ** 2 / poly.area < max_complexity:
            out.append(xy)
        if len(out) == n:
            break
    return out


def coarse(city, shape):
    """只算 IoU 的快速粗搜，回傳位置互不重疊的前幾名 (iou, cx, cy, size, ang)。"""
    x0, y0, x1, y1 = city.blocks.total_bounds
    cands = []
    for size in SIZES:
        for cx in np.arange(x0 + size / 2, x1 - size / 2, STEP):
            for cy in np.arange(y0 + size / 2, y1 - size / 2, STEP):
                for ang in ANGLES:
                    s, _ = city.score(city.place(shape, cx, cy, size, ang))
                    if s > 0.4:
                        cands.append((s, cx, cy, size, ang))
    cands.sort(reverse=True)
    seeds = []
    for c in cands:
        if all(far_apart(c, p) for p in seeds):
            seeds.append(c)
        if len(seeds) == SEEDS:
            break
    return seeds


def evaluate(city, shape, t, k):
    """回傳 (總分, 信心, IoU, 路線)。總分 = 0.7×分類器信心 + 0.3×IoU − 鋸齒懲罰"""
    pts = city.place(shape, *t)
    iou, chosen = city.score(pts)
    poly = route_of(city, chosen) if iou > 0.4 else None
    if poly is None:
        return -1, 0, iou, None
    p = predict(NET, render(poly.exterior.coords)[None])[0, k]
    zig = ZIGZAG * max(0.0, poly.exterior.length / Polygon(pts).exterior.length - 1.05)
    return 0.7 * p + 0.3 * iou - zig, p, iou, poly


def search_class(city, c, rng):
    k = K[c]
    cands = []
    for shape in templates(c):
        for _, *t in coarse(city, shape):
            j, p, iou, poly = evaluate(city, shape, t, k)
            if poly is not None:
                cands.append([j, p, iou, poly, shape, t])
    cands.sort(key=lambda r: -r[0])
    # 前幾名隨機小步爬山
    for r in cands[:4]:
        for _ in range(REFINE):
            t = r[5]
            t2 = [t[0] + rng.normal(0, 30), t[1] + rng.normal(0, 30), t[2] * math.exp(rng.normal(0, .05)), t[3] + rng.normal(0, 3)]
            j, p, iou, poly = evaluate(city, r[4], t2, k)
            if j > r[0]:
                r[:4], r[5] = [j, p, iou, poly], t2
    cands.sort(key=lambda r: -r[0])
    return cands


def recognize_old(name):
    """把 shapes.py 用手畫範本找到的路線也拿給分類器看，看認不認得出來。"""
    f = Path("out") / name / "results.json"
    if not f.exists():
        return []
    out = []
    for r in json.loads(f.read_text(encoding="utf-8")):
        ll = np.array(r["coords"])
        xy = np.c_[ll[:, 1] * math.cos(math.radians(ll[:, 0].mean())), ll[:, 0]]
        p = predict(NET, render(xy)[None])[0]
        top = np.argsort(-p)[:3]
        out.append({"template": r["template"], "rank": r["rank"], "guess": [[ZH[CLASSES[i]], round(float(p[i]), 3)] for i in top]})
    return out


def main(name):
    rng = np.random.default_rng(0)
    city = City(name)
    out = Path("out") / name
    (out / "discover").mkdir(parents=True, exist_ok=True)
    (out / "recognize.json").write_text(json.dumps(recognize_old(name), ensure_ascii=False, indent=1), encoding="utf-8")

    found = []
    for c in CLASSES:
        t0 = time.time()
        cands = search_class(city, c, rng)
        if not cands:
            print(f"{ZH[c]}：找不到")
            continue
        j, p, iou, poly, shape, t = cands[0]
        probs = predict(NET, render(poly.exterior.coords)[None])[0]
        found.append({"class": c, "zh": ZH[c], "score": float(j), "conf": float(p), "iou": float(iou),
                      "top1": ZH[CLASSES[int(probs.argmax())]], "km": poly.exterior.length / 1000, "poly": poly, "shape": shape, "t": t})
        print(f"{ZH[c]}：信心 {p:.0%}、IoU {iou:.2f}、{poly.exterior.length / 1000:.1f} km、分類器第一名猜 {found[-1]['top1']}"
              f"（{time.time() - t0:.0f} 秒）")

    found.sort(key=lambda r: -r["score"])
    streets = gpd.read_file(f"data/{name}_streets.geojson")
    n = min(12, len(found))
    fig, axes = plt.subplots(math.ceil(n / 4), 4, figsize=(16, 4.3 * math.ceil(n / 4)))
    js = []
    for i, r in enumerate(found):
        poly = r["poly"]
        ll = gpd.GeoSeries([poly], crs=city.blocks.crs).to_crs(4326).iloc[0]
        lonlat = list(ll.exterior.coords)
        (out / "discover" / f"{r['class']}.gpx").write_text(to_gpx(r["zh"], lonlat), encoding="utf-8")
        js.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("class", "zh", "score", "conf", "iou", "top1", "km")}
                  | {"coords": [[round(la, 6), round(lo, 6)] for lo, la in lonlat]})
        if i >= n:
            continue
        ax = axes.flat[i]
        cx, cy, size, _ = r["t"]
        pad = size * 0.7
        ax.set_xlim(cx - pad, cx + pad)
        ax.set_ylim(cy - pad, cy + pad)
        streets.plot(ax=ax, color="#ced4da", lw=.5)
        ax.fill(*poly.exterior.xy, color="#ffe8cc")
        ax.plot(*poly.exterior.xy, color="#e8590c", lw=2.2)
        ins = ax.inset_axes([0.74, 0.74, 0.25, 0.25])  # 右上角：分類器看到的 28×28 圖
        ins.imshow(render(poly.exterior.coords), cmap="gray_r")
        ins.set_xticks([])
        ins.set_yticks([])
        ax.set_title(f"{r['zh']}　信心 {r['conf']:.0%}　IoU {r['iou']:.2f}　{r['km']:.1f} km", fontsize=12)
    for ax in axes.flat:
        ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out / "discover.png", dpi=80)
    (out / "discover.json").write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
    print(f"完成：{out / 'discover.png'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "daan")
