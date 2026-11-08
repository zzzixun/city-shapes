"""在城市街廓裡找出最像某個圖案的跑步路線。

做法（以「面」貼合，而不是以「線」貼合）：
  1. 把圖案放到地圖上（位置 × 大小 × 角度）
  2. 選出「有一半以上面積落在圖案裡」的街廓，合併成一塊
  3. 這塊的外輪廓全部沿著街道，就是一條跑得出來的封閉路線
  4. 評分 = IoU（合併區塊與圖案的交集 ÷ 聯集）− 鋸齒懲罰（路線比圖案輪廓長太多＝一直鑽小巷轉彎）
     粗搜全部擺法（只算 IoU，很快）→ 挑位置不重疊的前幾名 → 用完整評分局部微調

用法：python shapes.py daan
輸出：out/<名稱>/  每個圖案前三名的 GPX、預覽圖 gallery.png、results.json
"""
import json
import math
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.ops import unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]

RES = 10  # 網格解析度（公尺）
SIZES = [700, 1000, 1400]  # 圖案寬度（公尺）
ANGLES = range(-30, 31, 10)  # 旋轉角度（度）
STEP = 100  # 粗搜的位置間距（公尺）
ZIGZAG = 0.5  # 鋸齒懲罰權重：路線長度每超過圖案輪廓 10%，扣 0.05


def heart(n=40):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = 16 * np.sin(t) ** 3, 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return np.c_[x, y] / 17


def star():
    a = np.pi / 2 + np.arange(10) * np.pi / 5
    r = np.where(np.arange(10) % 2, 0.42, 1)
    return np.c_[r * np.cos(a), r * np.sin(a)]


# 和 GPS Art 網頁版相同的範本（座標約在 [-1, 1]，y 朝上）
TEMPLATES = {
    "愛心": heart(),
    "星星": star(),
    "臘腸狗": np.array([[-.8, .15], [.5, .15], [.65, .4], [.8, .45], [1, .25], [.9, .1], [.7, .05], [.7, -.25], [.55, -.25], [.55, -.1],
                      [-.55, -.1], [-.55, -.25], [-.7, -.25], [-.75, .05], [-1, .35]]),
    "魚": np.array([[-.6, 0], [-.2, .4], [.4, .45], [1, 0], [.4, -.45], [-.2, -.4], [-.6, 0], [-1, -.4], [-1, .4]]),
    "房子": np.array([[-.8, -1], [-.8, .2], [0, 1], [.8, .2], [.8, -1]]),
    "箭頭": np.array([[-1, -.2], [.3, -.2], [.3, -.6], [1, 0], [.3, .6], [.3, .2], [-1, .2]]),
}


class City:
    """把街廓畫成編號網格，方便快速算「圖案蓋住哪些街廓」。"""

    def __init__(self, name):
        self.blocks = gpd.read_file(f"data/{name}_blocks.geojson")
        self.geoms = list(self.blocks.geometry)
        x0, y0, x1, y1 = self.blocks.total_bounds
        self.x0, self.y1 = x0, y1
        self.W, self.H = int((x1 - x0) / RES) + 1, int((y1 - y0) / RES) + 1
        img = Image.new("I", (self.W, self.H), 0)
        d = ImageDraw.Draw(img)
        for i, g in enumerate(self.geoms):
            d.polygon([self.px(x, y) for x, y in g.exterior.coords], fill=i + 1)
        self.ids = np.asarray(img, dtype=np.int32)
        self.pix = np.bincount(self.ids.ravel(), minlength=len(self.geoms) + 1)
        print(f"{len(self.geoms)} 個街廓，網格 {self.W}×{self.H}")

    def px(self, x, y):
        return ((x - self.x0) / RES, (self.y1 - y) / RES)

    def place(self, shape, cx, cy, size, ang):
        """圖案擺到地圖上，回傳公尺座標的頂點。"""
        a = math.radians(ang)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return shape @ R.T * (size / 2) + [cx, cy]

    def score(self, pts):
        """回傳 (IoU, 被選中的街廓編號)。只在圖案的外框範圍內計算，速度快。"""
        P = np.array([self.px(x, y) for x, y in pts])
        c0, r0 = np.floor(P.min(0)).astype(int) - 2
        c1, r1 = np.ceil(P.max(0)).astype(int) + 2
        if c0 < 0 or r0 < 0 or c1 > self.W or r1 > self.H:
            return 0.0, None
        img = Image.new("1", (c1 - c0, r1 - r0), 0)
        ImageDraw.Draw(img).polygon([(x - c0, y - r0) for x, y in P], fill=1)
        T = np.asarray(img)
        win = self.ids[r0:r1, c0:c1]
        cover = np.bincount(win[T], minlength=len(self.pix)) / np.maximum(self.pix, 1)
        cover[0] = 0
        chosen = cover > 0.5
        U = chosen[win]
        inter, union = (U & T).sum(), (U | T).sum()
        return (inter / union if union else 0.0), np.flatnonzero(chosen) - 1


def route_of(city, chosen):
    """被選中的街廓合併成一塊，取外輪廓當路線。先外擴再內縮 12 m，把分隔道路的縫補起來。"""
    g = unary_union([city.geoms[i] for i in chosen]).buffer(12, join_style=2).buffer(-12, join_style=2)
    if g.geom_type == "MultiPolygon":
        parts = sorted(g.geoms, key=lambda p: p.area, reverse=True)
        if parts[1].area > 0.02 * parts[0].area:
            return None  # 分成兩塊（例如魚尾巴沒接上身體），跑不成一圈
        g = parts[0]
    return Polygon(g.exterior).simplify(3)


def full_score(city, shape, t):
    """完整評分：IoU − 鋸齒懲罰。回傳 (分數, IoU, 路線多邊形, 選中的街廓)。"""
    pts = city.place(shape, *t)
    iou, chosen = city.score(pts)
    poly = route_of(city, chosen) if iou > 0.3 else None
    if poly is None:
        return 0.0, iou, None, chosen
    ratio = poly.exterior.length / Polygon(pts).exterior.length
    return iou - ZIGZAG * max(0.0, ratio - 1.05), iou, poly, chosen


def far_apart(a, b):
    return math.hypot(a[1] - b[1], a[2] - b[2]) > 0.6 * max(a[3], b[3])


def search(city, shape):
    x0, y0, x1, y1 = city.blocks.total_bounds
    cands = []
    for size in SIZES:
        for cx in np.arange(x0 + size / 2, x1 - size / 2, STEP):
            for cy in np.arange(y0 + size / 2, y1 - size / 2, STEP):
                for ang in ANGLES:
                    s, _ = city.score(city.place(shape, cx, cy, size, ang))
                    if s > 0.3:
                        cands.append((s, cx, cy, size, ang))
    cands.sort(reverse=True)
    # 先挑位置互不重疊的前 8 名，避免微調完全部擠在同一個地方
    seeds = []
    for c in cands:
        if all(far_apart(c, p) for p in seeds):
            seeds.append(c)
        if len(seeds) == 8:
            break
    # 用完整評分（含鋸齒懲罰）做隨機小步爬山
    rng = np.random.default_rng(0)
    best = []
    for _, *t in seeds:
        sc = full_score(city, shape, t)[0]
        for _ in range(60):
            t2 = [t[0] + rng.normal(0, 30), t[1] + rng.normal(0, 30), t[2] * math.exp(rng.normal(0, .05)), t[3] + rng.normal(0, 3)]
            s2 = full_score(city, shape, t2)[0]
            if s2 > sc:
                sc, t = s2, t2
        best.append((sc, *t))
    best.sort(reverse=True)
    picked = []
    for b in best:
        if all(far_apart(b, p) for p in picked):
            picked.append(b)
        if len(picked) == 3:
            break
    return picked, len(cands)


def to_gpx(name, lonlat):
    pts = "\n".join(f'<trkpt lat="{la:.6f}" lon="{lo:.6f}"></trkpt>' for lo, la in lonlat)
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="city-shapes" xmlns="http://www.topografix.com/GPX/1/1">'
            f"<trk><name>{name}</name><trkseg>\n{pts}\n</trkseg></trk></gpx>")


def main(name):
    city = City(name)
    out = Path("out") / name
    out.mkdir(parents=True, exist_ok=True)
    streets = gpd.read_file(f"data/{name}_streets.geojson")
    results = []
    fig, axes = plt.subplots(len(TEMPLATES), 3, figsize=(12, 4 * len(TEMPLATES)))
    for row, (tname, shape) in enumerate(TEMPLATES.items()):
        t0 = time.time()
        picked, n = search(city, shape)
        print(f"{tname}：{n} 個擺法超過門檻，{time.time() - t0:.0f} 秒")
        for k in range(3):
            ax = axes[row][k]
            ax.set_axis_off()
            if k >= len(picked):
                continue
            s, cx, cy, size, ang = picked[k]
            tpl = city.place(shape, cx, cy, size, ang)
            s, iou, poly, chosen = full_score(city, shape, (cx, cy, size, ang))
            if poly is None:
                continue
            ll = gpd.GeoSeries([poly], crs=city.blocks.crs).to_crs(4326).iloc[0]
            lonlat = list(ll.exterior.coords)
            km = poly.exterior.length / 1000
            fn = f"{tname}-{k + 1}.gpx"
            (out / fn).write_text(to_gpx(f"{tname} #{k + 1}", lonlat), encoding="utf-8")
            results.append({"template": tname, "rank": k + 1, "score": round(float(s), 3), "iou": round(float(iou), 3), "km": round(km, 2),
                            "coords": [[round(la, 6), round(lo, 6)] for lo, la in lonlat]})
            # 預覽：灰色街道、淡橘色選中的街廓、橘色路線、藍色虛線是原圖案
            pad = size * 0.7
            ax.set_xlim(cx - pad, cx + pad)
            ax.set_ylim(cy - pad, cy + pad)
            streets.plot(ax=ax, color="#ced4da", lw=.5)
            ax.fill(*poly.exterior.xy, color="#ffe8cc")
            ax.plot(*poly.exterior.xy, color="#e8590c", lw=2.2)
            ax.plot(*np.vstack([tpl, tpl[:1]]).T, color="#1c7ed6", lw=1.2, ls="--")
            ax.set_title(f"{tname} #{k + 1}　IoU {iou:.2f}　分數 {s:.2f}　{km:.1f} km", fontsize=11)
    plt.tight_layout()
    fig.savefig(out / "gallery.png", dpi=80)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"完成：{out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "daan")
