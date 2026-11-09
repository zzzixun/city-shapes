"""檢查分類器信心是不是被「搜尋」灌水了。

discover.py 是直接把分類器信心當目標去最佳化，分數一定偏高（模型可能被特定的鋸齒騙到）。這裡做兩個獨立檢查：
  1. 抖動：路線圖小幅旋轉 / 縮放 / 改線寬後重新分類，取平均。真的像的圖，信心不該掉太多
  2. 對照組：隨機橢圓擺在城市裡圈出普通的街廓塊，看分類器多常給出高信心（等於誤報率）

用法：python validate.py daan xinyi
輸出：out/validate.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from qd import CLASSES, ZH, load_model, predict, render
from shapes import City, route_of

sys.stdout.reconfigure(encoding="utf-8")
NET = load_model()
K = {c: i for i, c in enumerate(CLASSES)}
rng = np.random.default_rng(1)


def local_xy(coords):
    ll = np.array(coords)
    return np.c_[ll[:, 1] * math.cos(math.radians(ll[:, 0].mean())) * 111320, ll[:, 0] * 110540]


def jitter(xy, n=12):
    """回傳 n 張抖動過的 28×28 圖：旋轉 ±10°、長寬各 ±10%、線寬 10–18。"""
    imgs = []
    for _ in range(n):
        a = math.radians(rng.uniform(-10, 10))
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        v = (xy - xy.mean(0)) @ R.T * rng.uniform(0.9, 1.1, 2)
        imgs.append(render(v, width=int(rng.integers(10, 19))))
    return np.stack(imgs)


def ellipse(n=40):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[np.cos(t), rng.uniform(0.4, 1) * np.sin(t)]


def null_test(city, n=200):
    """隨機橢圓 → 合併街廓 → 分類器最高信心"""
    x0, y0, x1, y1 = city.blocks.total_bounds
    best = []
    while len(best) < n:
        t = (rng.uniform(x0 + 600, x1 - 600), rng.uniform(y0 + 600, y1 - 600), rng.uniform(800, 1200), rng.uniform(0, 180))
        iou, chosen = city.score(city.place(ellipse(), *t))
        poly = route_of(city, chosen) if iou > 0.4 else None
        if poly is not None:
            p = predict(NET, render(poly.exterior.coords)[None])[0]
            best.append((float(p.max()), CLASSES[int(p.argmax())]))
    return best


def main(names):
    report = {}
    for name in names:
        found = json.loads((Path("out") / name / "discover.json").read_text(encoding="utf-8"))
        rows = []
        for r in found:
            p = predict(NET, jitter(local_xy(r["coords"])))
            rows.append({"zh": r["zh"], "conf": r["conf"], "jitter_conf": round(float(p[:, K[r["class"]]].mean()), 3),
                         "jitter_top1": round(float((p.argmax(1) == K[r["class"]]).mean()), 3)})
        null = null_test(City(name))
        conf = np.array([b[0] for b in null])
        counts = {}
        for c, cl in null:
            if c > 0.9:
                counts[ZH[cl]] = counts.get(ZH[cl], 0) + 1
        report[name] = {"found": rows, "null_median": round(float(np.median(conf)), 3),
                        "null_over_90": round(float((conf > 0.9).mean()), 3), "null_over_90_classes": counts}
        print(f"\n{name}：搜尋出的路線抖動後平均信心 {np.mean([x['jitter_conf'] for x in rows]):.0%}"
              f"（原本 {np.mean([x['conf'] for x in rows]):.0%}）")
        for x in sorted(rows, key=lambda x: -x["jitter_conf"]):
            print(f"  {x['zh']}：{x['conf']:.0%} → {x['jitter_conf']:.0%}，抖動後仍猜對 {x['jitter_top1']:.0%}")
        print(f"  對照組（隨機橢圓圈出的街廓）：最高信心中位數 {np.median(conf):.0%}，超過 90% 的比例 {(conf > 0.9).mean():.0%} {counts}")
    Path("out/validate.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["daan", "xinyi"])
