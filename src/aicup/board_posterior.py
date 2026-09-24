"""板面 CNN 的題目層級後驗——校準端與推論端共用的唯一實作。

**為什麼要獨立成一個模組**：這個成分有一個曾經造成 0.198 Spearman 損失的
陷阱（RESULTS.md §2.27）——集合的成員必須是**每一個 snapshot**，
不能先把一局的 S 個 snapshot 平均。訓練時 `aicup.dataset.QuestionDataset`
用的是前者；只要有任何一條評估路徑用了後者，餵給 trunk 的就是一張
「平均局面」，一種它從沒見過的輸入分布。

先前正是因為這個邏輯被抄在三個地方（dataset / train / eval_blend），
修正只改到兩處。現在只有這裡一份，`predict.py` 與 `eval_blend.py` 都呼叫它。
"""

from __future__ import annotations

import numpy as np
import torch

from .losses import corn_to_proba
from .planes import game_planes, n_planes, planes_to_float
from .goboard import SIZE


def load_board_net(path: str, device) -> tuple:
    """讀 checkpoint，回傳 (net, n_snapshots)。一律走 strict=True 的 `load_gonet`。"""
    from .nets import load_gonet
    ck = torch.load(path, map_location=device, weights_only=False)
    net = load_gonet(ck["model"], ck.get("config"))
    net.eval().to(device)
    return net, int(ck.get("args", {}).get("snapshots", 4))


def question_members(pairs, n_snapshots: int) -> np.ndarray:
    """(sgf, color) 的清單 → 一題的集合成員張量 (N, C, 19, 19)。

    ⚠️ 每個 snapshot 各自是一個成員，**不要先平均**（見模組說明）。
    """
    members: list[np.ndarray] = []
    for sgf, color in pairs:
        t = game_planes(sgf, color, n_snapshots=n_snapshots)
        if t is not None:
            members.extend(planes_to_float(t))
    if not members:
        members = [np.zeros((n_planes(), SIZE, SIZE), dtype=np.float32)]
    return np.stack(members)


def dihedral(x: np.ndarray, k: int) -> np.ndarray:
    """棋盤的 8 種對稱之一（k=0 是恆等）。作用在最後兩軸。"""
    if k >= 4:
        x = np.flip(x, axis=-1)
    return np.rot90(x, k % 4, axes=(-2, -1))


@torch.no_grad()
def board_question_posteriors(net, n_snapshots: int, groups, device,
                              tta: int = 1) -> np.ndarray:
    """一組題目（每題是 (sgf, color) 的清單）→ (Q, 10) 的後驗機率。

    與其他成分一樣，softmax 頭與 CORN 頭各半——CORN 帶序數結構、
    softmax 不受單調假設拘束，兩者平均在每個成分上都測過比單獨好。

    `tta` = 使用前幾種棋盤對稱（1 = 不做、2 = 加左右翻轉、8 = 全部）。
    訓練時有做 8 種對稱的增強，所以每個版本都是模型見過的分布；
    各版本在整題層級池化後取幾何平均。探針（1,500 題）：
    Spearman 0.9210 → 0.9243（tta=8），5/10/15/20 局四種題型都提升。
    代價是前向次數乘上 tta（重播只做一次）。
    """
    order = (0, 4, 2, 6, 1, 3, 5, 7)          # 先恆等與翻轉，tta=2 就是左右翻轉
    ks = order[:max(1, min(8, int(tta)))]
    out = []
    for pairs in groups:
        m = question_members(pairs, n_snapshots)
        x = np.stack([np.ascontiguousarray(dihedral(m, k)) for k in ks])
        o = net.forward_set(torch.from_numpy(x).float().to(device))
        pc = corn_to_proba(o["corn_logits"]).cpu().numpy()
        pc = pc / np.maximum(pc.sum(axis=1, keepdims=True), 1e-12)
        p = 0.5 * torch.softmax(o["rank_logits"], -1).cpu().numpy() + 0.5 * pc
        lg = np.log(np.clip(p, 1e-12, None)).mean(axis=0)
        p = np.exp(lg - lg.max())
        out.append(p / p.sum())
    return np.stack(out)


def groups_from_meta(meta, game_rows, cache) -> list:
    """校準端：題目的 meta 列索引 → (sgf, color) 清單。"""
    return [[(cache.get(int(meta.iloc[int(r)]["file_idx"]),
                        int(meta.iloc[int(r)]["row"])),
              meta.iloc[int(r)]["color"]) for r in rows] for rows in game_rows]


def groups_from_flat(flat_sgf, flat_color, sizes) -> list:
    """推論端：扁平的 SGF/執色清單 ＋ 每題局數 → (sgf, color) 清單。"""
    groups, i = [], 0
    for n in sizes:
        groups.append(list(zip(flat_sgf[i:i + n], flat_color[i:i + n])))
        i += n
    return groups
