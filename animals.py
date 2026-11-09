"""實驗：比較複雜的動物輪廓，放大到多大才畫得像？

圖案取自 Quick, Draw! 手繪圖的外輪廓（每類 2 張），在整個臺北市用 lines.py 的方法，
分別限制圖寬 1 / 2 / 3 / 4 km 找最好的路線，比較：
  - 相對誤差 =（漏畫 + 亂繞）/ 2 ÷ 圖寬
  - 尖角保留率
  - 分類器（獨立評審，沒拿來最佳化）認不認得出來

用法：python animals.py taipei      （需先跑 fetch.py "Taipei, Taiwan" taipei）
輸出：out/<名稱>/animals.png、animals.json、animals/*.gpx
"""
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from shapely.geometry import LineString

from discover import templates
from lines import Streets, search
from qd import CLASSES, ZH, load_model, predict, render
from shapes import to_gpx

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]

ANIMALS = ["cat", "rabbit", "elephant", "duck", "whale", "snail", "bird", "dog", "shark", "airplane"]
WIDTHS = [1000, 2000, 3000, 4000]
NET = load_model()


def main(name):
    st = Streets(name)
    out = Path("out") / name
    (out / "animals").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(ANIMALS), len(WIDTHS), figsize=(4 * len(WIDTHS), 4 * len(ANIMALS)))
    js = []
    for row, c in enumerate(ANIMALS):
        tpls = templates(c, n=2, max_complexity=60)
        for col, w in enumerate(WIDTHS):
            t0 = time.time()
            best = None
            for shape in tpls:
                res = search(st, shape, sizes=[w])
                if res and (best is None or res[0][0] < best[0]):
                    best = res[0]
            ax = axes[row][col]
            ax.set_axis_off()
            if best is None:
                continue
            s, miss, wander, kept, route, tpl, (cx, cy, size, ang) = best
            p = predict(NET, render(route)[None])[0]
            km = LineString(route).length / 1000
            rel = (miss + wander) / 2 / w
            ll = gpd.GeoSeries([LineString(route)], crs=st.gdf.crs).to_crs(4326).iloc[0]
            (out / "animals" / f"{c}-{w // 1000}km.gpx").write_text(to_gpx(f"{ZH[c]} {w // 1000} km", list(ll.coords)), encoding="utf-8")
            js.append({"class": c, "zh": ZH[c], "width_m": w, "score": round(s, 2), "miss_m": round(miss, 1), "wander_m": round(wander, 1),
                       "rel_err": round(float(rel), 4), "corners": round(float(kept), 2), "km": round(km, 2),
                       "conf": round(float(p[CLASSES.index(c)]), 3), "top1": ZH[CLASSES[int(p.argmax())]],
                       "coords": [[round(la, 6), round(lo, 6)] for lo, la in ll.coords]})
            print(f"{ZH[c]} 寬 {w // 1000} km：相對誤差 {rel:.1%}、尖角 {kept:.0%}、跑 {km:.1f} km、"
                  f"分類器 {p[CLASSES.index(c)]:.0%}（猜 {js[-1]['top1']}）（{time.time() - t0:.0f} 秒）")
            pad = size * 0.65
            ax.set_xlim(cx - pad, cx + pad)
            ax.set_ylim(cy - pad, cy + pad)
            st.gdf.plot(ax=ax, color="#dee2e6", lw=.4)
            ax.plot(*np.vstack([tpl, tpl[:1]]).T, color="#1c7ed6", lw=1, ls="--")
            ax.plot(*route.T, color="#e8590c", lw=2)
            ax.set_title(f"{ZH[c]}　寬 {w // 1000} km　跑 {km:.1f} km\n誤差 {rel:.1%}　尖角 {kept:.0%}　分類器 {p[CLASSES.index(c)]:.0%}",
                         fontsize=10)
    plt.tight_layout()
    fig.savefig(out / "animals.png", dpi=70)
    (out / "animals.json").write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")

    print("\n各寬度平均：")
    for w in WIDTHS:
        r = [x for x in js if x["width_m"] == w]
        print(f"  {w // 1000} km：相對誤差 {np.mean([x['rel_err'] for x in r]):.1%}、尖角 {np.mean([x['corners'] for x in r]):.0%}、"
              f"跑 {np.mean([x['km'] for x in r]):.1f} km、分類器認對 {sum(x['top1'] == x['zh'] for x in r)}/{len(r)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "taipei")
