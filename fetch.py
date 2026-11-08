"""下載行政區的街道路網，切成街廓（被街道圍起來的區塊）。

用法：python fetch.py "Da'an District, Taipei, Taiwan" daan
輸出：data/<名稱>_blocks.geojson（公尺座標）、data/<名稱>_streets.geojson
"""
import sys
from pathlib import Path

import geopandas as gpd
import osmnx as ox
from shapely.ops import polygonize, unary_union

sys.stdout.reconfigure(encoding="utf-8")

# 跑步用得到的街道：台北很多「弄」標成 service=alley，要算進來；停車場通道、車道出入口不算
STREETS = ('["highway"~"trunk|primary|secondary|tertiary|unclassified|residential|living_street|pedestrian|_link|service"]'
           '["service"!~"parking_aisle|driveway|drive-through"]["access"!~"private"]')
MIN_AREA, MAX_AREA = 300, 150000  # m²：太小是路口分隔島；大安森林公園約 26 萬 m² 不算
MIN_WIDTH = 16  # m：比這窄的是分隔島或雙向分隔道路中間的長條


def main(place, name):
    out = Path("data")
    out.mkdir(exist_ok=True)
    print(f"下載 {place} 的路網…")
    # 行政區外擴 300 m 一起下載，邊界上的街廓才完整
    area = ox.geocode_to_gdf(place)
    poly = area.to_crs(area.estimate_utm_crs()).buffer(300).to_crs(4326).iloc[0]
    G = ox.graph_from_polygon(poly, custom_filter=STREETS, simplify=True, retain_all=False)
    G = ox.project_graph(G)  # 轉成公尺座標（UTM）
    edges = ox.graph_to_gdfs(G, nodes=False)
    print(f"街道 {len(edges)} 段，座標系 {edges.crs}")

    blocks = [p for p in polygonize(unary_union(list(edges.geometry)))
              if MIN_AREA <= p.area <= MAX_AREA and not p.buffer(-MIN_WIDTH / 2).is_empty]
    gdf = gpd.GeoDataFrame(geometry=blocks, crs=edges.crs)
    gdf.to_file(out / f"{name}_blocks.geojson", driver="GeoJSON")
    edges[["geometry"]].to_file(out / f"{name}_streets.geojson", driver="GeoJSON")
    print(f"街廓 {len(gdf)} 塊，面積中位數 {gdf.area.median():.0f} m²")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Da'an District, Taipei, Taiwan",
         sys.argv[2] if len(sys.argv) > 2 else "daan")
