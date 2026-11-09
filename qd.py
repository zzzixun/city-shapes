"""Quick, Draw! 分類器的共用部分：類別、模型、把路線畫成 28×28 圖、手繪圖的風格轉換。"""
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from torch import nn

# 挑「用一圈外輪廓就認得出來」的類別
CLASSES = ["fish", "whale", "shark", "bird", "duck", "cat", "dog", "rabbit", "elephant", "snail", "star", "house",
           "apple", "pear", "mushroom", "cloud", "lightning", "t-shirt", "sock", "hand", "airplane", "crown", "light bulb", "moon"]
ZH = {"fish": "魚", "whale": "鯨魚", "shark": "鯊魚", "bird": "鳥", "duck": "鴨子", "cat": "貓", "dog": "狗", "rabbit": "兔子",
      "elephant": "大象", "snail": "蝸牛", "star": "星星", "house": "房子", "apple": "蘋果", "pear": "西洋梨", "mushroom": "香菇",
      "cloud": "雲", "lightning": "閃電", "t-shirt": "T 恤", "sock": "襪子", "hand": "手", "airplane": "飛機", "crown": "皇冠",
      "light bulb": "燈泡", "moon": "月亮"}
DATA = "cache/quickdraw"
MODEL = "models/qd_cnn.pt"


class Net(nn.Module):
    """很小的 CNN（約 40 萬參數），CPU 幾分鐘就能訓練完。"""

    def __init__(self, n=len(CLASSES)):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 9, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n))

    def forward(self, x):
        return self.f(x)


def load_model():
    net = Net()
    net.load_state_dict(torch.load(MODEL, map_location="cpu"))
    return net.eval()


@torch.no_grad()
def predict(net, imgs):
    """imgs: (N, 28, 28) uint8 → (N, 類別數) 機率"""
    x = torch.from_numpy(np.asarray(imgs, dtype=np.float32)[:, None] / 255)
    return torch.softmax(net(x), 1).numpy()


def render(xy, width=14):
    """把一圈路線（公尺座標、y 朝上）畫成 Quick, Draw! 風格：置中、撐滿 28×28、線寬約 2 像素。"""
    xy = np.asarray(xy, float)
    xy = xy - (xy.min(0) + xy.max(0)) / 2
    xy = xy / np.abs(xy).max() * 108 * [1, -1] + 128  # 256 畫布，左右留白和 Quick, Draw! 一樣
    img = Image.new("L", (256, 256), 0)
    pts = [tuple(p) for p in xy]
    ImageDraw.Draw(img).line(pts + pts[:1], fill=255, width=width, joint="curve")
    return np.asarray(img.resize((28, 28), Image.BOX))


def silhouette(img):
    """手繪圖 → 實心剪影（把小缺口補起來再填滿內部）。"""
    m = ndimage.binary_closing(img > 40, np.ones((3, 3)), iterations=2)
    return ndimage.binary_fill_holes(m)


def outline_style(img, block=1):
    """手繪圖 → 只剩外輪廓的版本，block>1 時先變成方塊（模仿街廓的鋸齒）。訓練時當資料擴增，縮小風格差異。"""
    s = silhouette(img)
    if block > 1:
        s = s.reshape(28 // block, block, 28 // block, block).mean((1, 3)) > 0.4
        s = s.repeat(block, 0).repeat(block, 1)
    edge = s & ~ndimage.binary_erosion(s, iterations=2)
    return (ndimage.gaussian_filter(edge.astype(float), 0.5) * 255 / 0.75).clip(0, 255).astype(np.uint8)
