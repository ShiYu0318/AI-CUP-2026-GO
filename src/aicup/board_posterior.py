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


@torch.no_grad()
def board_question_posteriors(net, n_snapshots: int, groups, device) -> np.ndarray:
    """一組題目（每題是 (sgf, color) 的清單）→ (Q, 10) 的後驗機率。

    與其他成分一樣，softmax 頭與 CORN 頭各半——CORN 帶序數結構、
    softmax 不受單調假設拘束，兩者平均在每個成分上都測過比單獨好。
    """
    out = []
    for pairs in groups:
        x = torch.from_numpy(question_members(pairs, n_snapshots)).float()
        o = net.forward_set(x.to(device)[None])
        pc = corn_to_proba(o["corn_logits"]).cpu().numpy()
        pc = pc / np.maximum(pc.sum(axis=1, keepdims=True), 1e-12)
        p = (0.5 * torch.softmax(o["rank_logits"], -1).cpu().numpy() + 0.5 * pc)[0]
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
