"""用「線」描圖案：沿著街道走出最貼近圖案外框的路線（Waschk & Krüger 2019 的簡化版）。

做法：
  1. 粗搜擺法：街道畫成 5 m 網格做距離轉換，圖案取樣點查「離最近道路多遠」取平均（和 GPS Art 網頁版相同）
  2. 沿街道找路：圖案上每隔約 150 m 取一個路標，相鄰路標之間跑 Dijkstra，
     路段成本 = 長度 ×（1 + (離圖案距離 / 25 m)²），越偏離圖案越貴，路線就會貼著圖案走
  3. 雙向評分：
       漏畫 = 圖案上的點到路線的平均距離
       亂繞 = 路線上的點到圖案的平均距離
       尖角 = 圖案的尖角（轉超過 40°）有幾成在路線 6% 圖寬以內
     分數（越小越好）= (漏畫 + 亂繞) / 2 ÷ 圖寬 × 100 + 20 × (1 − 尖角保留率)

用法：python lines.py daan
輸出：out/<名稱>/lines.png、lines.json、lines/*.gpx
"""
import json
import math
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import shapely
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, Point

from shapes import TEMPLATES, to_gpx

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]

RES = 5  # 距離網格解析度（公尺）
SIZES = [800, 1200, 1600, 2000]
ANGLES = range(-30, 31, 10)
STEP = 50
TOP = 5  # 每種大小實際找路的擺法數（不同大小分開挑，避免小尺寸因為絕對距離小而全部勝出）
SIGMA = 25  # 偏離圖案多遠成本變兩倍（公尺）
WAYPOINT = 150  # 路標間距（公尺）


def resample(ring, step):
    """封閉折線每隔 step 公尺取一點（ring 不重複起點）"""
    pts = np.vstack([ring, ring[:1]])
    seg = np.diff(pts, axis=0)
    L = np.hypot(*seg.T)
    out = []
    for p, d, l in zip(pts[:-1], seg, L):
        n = max(1, int(math.ceil(l / step)))
        out.append(p + d * (np.arange(n) / n)[:, None])
    return np.vstack(out)


def corners(ring, deg=40):
    """轉角超過 deg 的頂點"""
    a, b = ring - np.roll(ring, 1, 0), np.roll(ring, -1, 0) - ring
    ang = np.degrees(np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))))
    return ring[ang > deg]


def place(shape, cx, cy, size, ang):
    a = math.radians(ang)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return shape @ R.T * (size / 2) + [cx, cy]


class Streets:
    def __init__(self, name):
        self.gdf = gpd.read_file(f"data/{name}_streets.geojson")
        lines = [g for geom in self.gdf.geometry for g in getattr(geom, "geoms", [geom])]
        # 路網圖：每條街道的每一小段都是一條邊，端點四捨五入到 0.1 m 當節點
        key, xy, rows, cols, w = {}, [], [], [], []
        mids = []
        for ln in lines:
            c = np.asarray(ln.coords)[:, :2]
            for p, q in zip(c[:-1], c[1:]):
                ij = []
                for v in (p, q):
                    k = (round(v[0], 1), round(v[1], 1))
                    if k not in key:
                        key[k] = len(xy)
                        xy.append(v)
                    ij.append(key[k])
                rows.append(ij[0])
                cols.append(ij[1])
                w.append(np.hypot(*(q - p)))
                mids.append((p, q))
        self.xy = np.array(xy)
        self.u, self.v, self.len = np.array(rows), np.array(cols), np.array(w)
        self.seg = np.array(mids)  # (邊數, 2 端點, 2)
        self.tree = shapely.STRtree(shapely.points(self.xy))
        # 距離網格
        x0, y0, x1, y1 = self.gdf.total_bounds
        self.x0, self.y1 = x0, y1
        W, H = int((x1 - x0) / RES) + 1, int((y1 - y0) / RES) + 1
        img = Image.new("1", (W, H), 1)
        d = ImageDraw.Draw(img)
        for ln in lines:
            d.line([((x - x0) / RES, (y1 - y) / RES) for x, y in np.asarray(ln.coords)[:, :2]], fill=0, width=1)
        self.dt = ndimage.distance_transform_edt(np.asarray(img)) * RES
        self.bounds = (x0, y0, x1, y1)
        print(f"{len(self.xy)} 個節點、{len(self.len)} 段街道")

    def road_dist(self, pts):
        c = ((pts[..., 0] - self.x0) / RES).astype(int)
        r = ((self.y1 - pts[..., 1]) / RES).astype(int)
        ok = (c >= 0) & (r >= 0) & (c < self.dt.shape[1]) & (r < self.dt.shape[0])
        return np.where(ok, self.dt[r.clip(0, self.dt.shape[0] - 1), c.clip(0, self.dt.shape[1] - 1)], 999)

    def route(self, tpl):
        """沿街道描出 tpl（封閉折線），回傳路線座標 (N, 2)"""
        ring = LineString(np.vstack([tpl, tpl[:1]]))
        # 每段街道離圖案的距離（兩端點與中點平均）
        p, q = self.seg[:, 0], self.seg[:, 1]
        d = sum(shapely.distance(shapely.points(v), ring) for v in (p, q, (p + q) / 2)) / 3
        w = self.len * (1 + (d / SIGMA) ** 2) + 1e-3
        n = len(self.xy)
        G = csr_matrix((np.r_[w, w], (np.r_[self.u, self.v], np.r_[self.v, self.u])), shape=(n, n))
        wps = resample(tpl, WAYPOINT)
        nodes = [int(self.tree.nearest(Point(x, y))) for x, y in wps]
        nodes = [a for i, a in enumerate(nodes) if a != nodes[i - 1]] or nodes[:1]
        path = [nodes[0]]
        for a, b in zip(nodes, nodes[1:] + nodes[:1]):
            dist, pred = dijkstra(G, indices=a, return_predecessors=True, limit=WAYPOINT * 20)
            if not np.isfinite(dist[b]):
                return None
            seg = [b]
            while seg[-1] != a:
                seg.append(pred[seg[-1]])
            path += seg[::-1][1:]
        # 去掉「走進去又原路退出來」的小尖刺
        out = []
        for k in path:
            if len(out) >= 2 and out[-2] == k:
                out.pop()
            else:
                out.append(k)
        return self.xy[out]


def evaluate(route, tpl, size):
    """回傳 (分數, 漏畫 m, 亂繞 m, 尖角保留率)"""
    ring = LineString(np.vstack([tpl, tpl[:1]]))
    rl = LineString(route)
    miss = shapely.distance(shapely.points(resample(tpl, 10)), rl).mean()
    wander = shapely.distance(shapely.points(resample(route[:-1], 10)), ring).mean()
    cs = corners(tpl)
    kept = (shapely.distance(shapely.points(cs), rl) < 0.06 * size).mean() if len(cs) else 1.0
    return (miss + wander) / 2 / size * 100 + 20 * (1 - kept), miss, wander, kept


def search(st, shape):
    x0, y0, x1, y1 = st.bounds
    picked = []
    for size in SIZES:
        cands = []
        unit = resample(shape * (size / 2), 25)
        cx, cy = np.meshgrid(np.arange(x0 + size / 2, x1 - size / 2, STEP), np.arange(y0 + size / 2, y1 - size / 2, STEP))
        C = np.c_[cx.ravel(), cy.ravel()]
        for ang in ANGLES:
            a = math.radians(ang)
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            pts = C[:, None, :] + (unit @ R.T)[None]
            s = st.road_dist(pts).mean(1)
            for i in np.argsort(s)[:200]:
                cands.append((s[i], C[i, 0], C[i, 1], size, ang))
        cands.sort()
        mine = []
        for c in cands:
            if all(math.hypot(c[1] - p[1], c[2] - p[2]) > 0.5 * c[3] for p in mine):
                mine.append(c)
            if len(mine) == TOP:
                break
        picked += mine
    res = []
    for _, cx, cy, size, ang in picked:
        tpl = place(shape, cx, cy, size, ang)
        r = st.route(tpl)
        if r is not None and len(r) > 3:
            res.append((*evaluate(r, tpl, size), r, tpl, (cx, cy, size, ang)))
    res.sort(key=lambda r: r[0])
    # 前幾名位置要分開（同一處不同大小只留最好的）
    out = []
    for r in res:
        c = r[6]
        if all(math.hypot(c[0] - o[6][0], c[1] - o[6][1]) > 0.4 * max(c[2], o[6][2]) for o in out):
            out.append(r)
    return out


def main(name):
    st = Streets(name)
    out = Path("out") / name
    (out / "lines").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(TEMPLATES), 3, figsize=(12, 4 * len(TEMPLATES)))
    js = []
    for row, (tname, shape) in enumerate(TEMPLATES.items()):
        t0 = time.time()
        res = search(st, shape)
        print(f"{tname}：{len(res)} 條路線（{time.time() - t0:.0f} 秒）"
              + "".join(f"\n  #{k + 1} 分數 {r[0]:.1f}、漏畫 {r[1]:.0f} m、亂繞 {r[2]:.0f} m、尖角 {r[3]:.0%}" for k, r in enumerate(res[:3])))
        for k in range(3):
            ax = axes[row][k]
            ax.set_axis_off()
            if k >= len(res):
                continue
            s, miss, wander, kept, route, tpl, (cx, cy, size, ang) = res[k]
            km = LineString(route).length / 1000
            ll = gpd.GeoSeries([LineString(route)], crs=st.gdf.crs).to_crs(4326).iloc[0]
            (out / "lines" / f"{tname}-{k + 1}.gpx").write_text(to_gpx(f"{tname} #{k + 1}", list(ll.coords)), encoding="utf-8")
            js.append({"template": tname, "rank": k + 1, "score": round(s, 2), "miss_m": round(miss, 1), "wander_m": round(wander, 1),
                       "corners": round(float(kept), 2), "km": round(km, 2), "coords": [[round(la, 6), round(lo, 6)] for lo, la in ll.coords]})
            pad = size * 0.7
            ax.set_xlim(cx - pad, cx + pad)
            ax.set_ylim(cy - pad, cy + pad)
            st.gdf.plot(ax=ax, color="#ced4da", lw=.5)
            ax.plot(*np.vstack([tpl, tpl[:1]]).T, color="#1c7ed6", lw=1.2, ls="--")
            ax.plot(*route.T, color="#e8590c", lw=2.2)
            ax.set_title(f"{tname} #{k + 1}　漏畫 {miss:.0f} m　亂繞 {wander:.0f} m　尖角 {kept:.0%}　{km:.1f} km", fontsize=10)
            ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out / "lines.png", dpi=80)
    (out / "lines.json").write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
    print(f"完成：{out / 'lines.png'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "daan")
