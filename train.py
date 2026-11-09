"""訓練 Quick, Draw! 分類器（只用 CPU）。

每類取 12,000 張訓練、2,000 張測試。訓練資料有 1/3 保留原圖、1/3 改成「只剩外輪廓」、1/3 改成「方塊化的外輪廓」，
讓模型看得懂街廓路線那種只有一圈、方方正正的畫法。

用法：python train.py
輸出：models/qd_cnn.pt、models/eval.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from qd import CLASSES, DATA, MODEL, Net, outline_style

sys.stdout.reconfigure(encoding="utf-8")
N_TRAIN, N_TEST, EPOCHS = 12000, 2000, 6
torch.set_num_threads(4)
torch.manual_seed(0)
rng = np.random.default_rng(0)


def load():
    X, Y = [], []
    for k, c in enumerate(CLASSES):
        a = np.load(f"{DATA}/{c}.npy", mmap_mode="r")
        idx = rng.choice(len(a), N_TRAIN + N_TEST, replace=False)
        X.append(np.array(a[np.sort(idx)]).reshape(-1, 28, 28)[rng.permutation(len(idx))])
        Y.append(np.full(len(idx), k))
    X, Y = np.stack(X), np.stack(Y)  # (類別, 張數, 28, 28)
    return (X[:, :N_TRAIN].reshape(-1, 28, 28), Y[:, :N_TRAIN].ravel(),
            X[:, N_TRAIN:].reshape(-1, 28, 28), Y[:, N_TRAIN:].ravel())


def restyle(X):
    """1/3 原圖、1/3 外輪廓、1/3 方塊化外輪廓（方塊 2 或 4 像素）"""
    X = X.copy()
    mode = rng.integers(0, 3, len(X))
    for i in np.flatnonzero(mode):
        X[i] = outline_style(X[i], 1 if mode[i] == 1 else int(rng.choice([2, 4])))
    return X


def accuracy(net, X, Y):
    net.eval()
    with torch.no_grad():
        p = torch.cat([net(torch.from_numpy(X[i:i + 4096, None].astype(np.float32) / 255)).argmax(1)
                       for i in range(0, len(X), 4096)]).numpy()
    return float((p == Y).mean())


def main():
    t0 = time.time()
    Xtr, Ytr, Xte, Yte = load()
    Xtr = restyle(Xtr)
    test = {"原圖": Xte, "外輪廓": np.stack([outline_style(x) for x in Xte]),
            "方塊外輪廓": np.stack([outline_style(x, 4) for x in Xte])}
    print(f"資料 {len(Xtr)} 張訓練、{len(Xte)} 張測試，準備 {time.time() - t0:.0f} 秒")

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps = EPOCHS * (len(Xtr) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=steps)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xt))
        for b in range(len(Xt) // 256):
            i = perm[b * 256:(b + 1) * 256]
            x = Xt[i, None].float() / 255
            loss = lossf(net(x), Yt[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        accs = {k: accuracy(net, v, Yte) for k, v in test.items()}
        print(f"第 {ep + 1} 輪  loss {loss.item():.3f}  測試正確率 " + "、".join(f"{k} {a:.1%}" for k, a in accs.items())
              + f"  （{time.time() - t0:.0f} 秒）")

    Path(MODEL).parent.mkdir(exist_ok=True)
    torch.save(net.state_dict(), MODEL)
    Path("models/eval.json").write_text(json.dumps({"classes": CLASSES, "accuracy": accs}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
