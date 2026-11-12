"""整理大台北掃描結果：每個圖案挑全區最好、位置不重疊的前 3 名，查出所在行政區，輸出總覽圖與 GPX。

掃描本身在瀏覽器裡跑（重用 GPS Art 網頁版的「指定範圍自動生成」，結果和網頁一致）：
  台北盆地每 0.025°（約 2.5 km）一個中心、半徑 2 km、目標 10 km，17 個範本各取該中心的第 1 名
用法：python scan_report.py <results.jsonl>
輸出：out/scan/gallery.png、out/scan/<圖案>-<名次>.gpx、out/scan/best.json
"""
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib
import numpy as np

from shapes import to_gpx

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Microsoft JhengHei", "Segoe UI Emoji", "sans-serif"]
OUT = Path("out/scan")


def centroid(c):
    a = np.array(c)
    return a[:, 0].mean(), a[:, 1].mean()


def km_between(p, q):
    return math.hypot((p[0] - q[0]) * 110.54, (p[1] - q[1]) * 111.32 * math.cos(math.radians(p[0])))


def district(lat, lon, zoom=14):
    """用 Nominatim 反查行政區（每秒最多 1 次）；落在公園、河邊時常查不到區名，改用較粗的層級再查一次"""
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat:.5f}&lon={lon:.5f}&zoom={zoom}&accept-language=zh-TW"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "city-shapes-research (github.com/zzzixun/city-shapes)"})
        with urllib.request.urlopen(req, timeout=20) as r:
            a = json.load(r).get("address", {})
        time.sleep(1.1)
        city = a.get("city") or a.get("county") or ""
        town = a.get("suburb") or a.get("town") or a.get("city_district") or a.get("district") or ""
        if not town and zoom > 10:
            return district(lat, lon, 10)
        return f"{city}{town}" if town not in city else city
    except Exception:
        return ""


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    by = {}
    for row in rows:
        for r in row["res"]:
            r["center"] = centroid(r["coords"])
            by.setdefault(r["k"], []).append(r)
    print(f"{len(rows)} 個中心點")
    OUT.mkdir(parents=True, exist_ok=True)
    best = {}
    for k, rs in by.items():
        rs.sort(key=lambda r: r["total"])
        picked = []
        for r in rs:
            if all(km_between(r["center"], p["center"]) > 1.5 for p in picked):
                picked.append(r)
            if len(picked) == 3:
                break
        best[k] = picked
    # 查行政區、輸出 GPX
    for k, picked in best.items():
        for i, r in enumerate(picked):
            r["area"] = district(*r["center"])
            name = k.split(" ", 1)[-1]
            (OUT / f"{name}-{i + 1}.gpx").write_text(to_gpx(f"{name} #{i + 1}（{r['area']}）", [(lo, la) for la, lo in r["coords"]]), encoding="utf-8")
        print(k, "｜", "；".join(f"{r['area']} {r['km']:.1f} km 分數 {r['total']:.1f}" for r in picked))
    # 總覽圖：每個圖案一列，前 3 名
    fig, axes = plt.subplots(len(best), 3, figsize=(10, 3.4 * len(best)))
    for row_axes, (k, picked) in zip(axes, best.items()):
        for ax, i in zip(row_axes, range(3)):
            ax.axis("off")
            if i >= len(picked):
                continue
            r = picked[i]
            kx = math.cos(math.radians(r["center"][0]))
            for T in r["tpl"]:
                T = np.array(T)  # 封閉的筆畫在網頁那邊已經接回起點
                ax.plot(T[:, 1] * kx, T[:, 0], "--", color="#1c7ed6", lw=.8)
            c = np.array(r["coords"])
            ax.plot(c[:, 1] * kx, c[:, 0], color="#e8590c", lw=1.4)
            ax.set_aspect("equal")
            ax.set_title(f"{k.split(' ', 1)[-1]} #{i + 1}　{r['area']}\n{r['km']:.1f} km　漏畫 {r['miss']:.0f} m　尖角 {r['kept']:.0%}", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / "gallery.png", dpi=60)
    json.dump({k: [{x: r[x] for x in ("area", "km", "total", "miss", "wander", "kept", "coords")} for r in v] for k, v in best.items()},
              open(OUT / "best.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"完成：{OUT / 'gallery.png'}")


if __name__ == "__main__":
    main(sys.argv[1])
