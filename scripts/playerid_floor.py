"""玩家辨識：本地驗證框架 + 手工特徵地板實驗。

完全複製官方測試協定：
  - 從「訓練集中完全沒用來訓練的玩家」裡取 400 人做候選庫（每人 100 局）
  - 每人另取 5~20 局（與候選庫不重疊）當成一道題
  - 交出 Top-5，用官方公式 e^-(r-1) 計分
這就是可以在 11/04 之前（沒有 leaderboard 的 67 天）用來判斷進步與否的離線指標。

⚠️ **2026-09-18 的早期探索腳本，已被函式庫取代，目前沒有任何程式引用它。**
留著是因為它是競賽一離線驗證協定的原始證據；
該協定現在長在 `src/aicup/benchmark.py` 的 `build_player_pool()`
與 `scripts/eval_player.py` 裡。結論見 RESULTS.md §1.3。
它不依賴 `src/aicup/`（只用 numpy/pandas），可以獨立重跑：
    python3 scripts/playerid_floor.py 4000
⚠️ 它的評估協定是**當時**的版本（題目局數分布 2026-09-20 才改成官方的
5/10/15/20），所以**絕對數字不能與現在的比較**——價值在方法論的證據。
"""
import glob, re, sys, numpy as np, pandas as pd
from collections import defaultdict

MOVE_RE = re.compile(r';([BW])\[([a-s])([a-s])\]')
rng = np.random.default_rng(7)
N_CAND = 400          # 候選庫人數，對齊官方
N_GALLERY = 100       # 每位候選人的局數，對齊官方
N_QUERY_MAX = 20      # 每題最多局數，對齊官方

# 盤面折疊到 8 重對稱的基本域 -> 55 格
FOLD = {}
_cells = []
for x in range(19):
    for y in range(19):
        a, b = min(x, 18 - x), min(y, 18 - y)
        a, b = min(a, b), max(a, b)
        if (a, b) not in FOLD:
            FOLD[(a, b)] = len(_cells); _cells.append((a, b))
NF = len(_cells)

def game_vec(sgf, color):
    mv = MOVE_RE.findall(sgf)
    if len(mv) < 30: return None
    pts = [(c, ord(x) - 97, ord(y) - 97) for c, x, y in mv]
    mine = [(i, x, y) for i, (c, x, y) in enumerate(pts) if c == color]
    if len(mine) < 15: return None

    def fold(x, y):
        a, b = min(x, 18 - x), min(y, 18 - y)
        return FOLD[(min(a, b), max(a, b))]
    def line(x, y): return min(x, y, 18 - x, 18 - y)

    op = np.zeros(NF); allm = np.zeros(NF)          # 布局репертуар / 全局分布
    for k, (i, x, y) in enumerate(mine):
        allm[fold(x, y)] += 1
        if k < 15: op[fold(x, y)] += 1
    op /= max(op.sum(), 1); allm /= max(allm.sum(), 1)

    sc = []
    sc.append(len(pts)); sc.append(len(mine))
    sc.append(1.0 if color == 'B' else 0.0)
    for lo, hi in [(0, 40), (40, 120), (120, 10 ** 9)]:
        sel = [(x, y) for i, x, y in mine if lo <= i < hi]
        if sel:
            ls = [line(x, y) for x, y in sel]
            sc += [np.mean(ls), np.std(ls)] + [np.mean([l == L for l in ls]) for L in range(5)]
            d = [abs(sel[k][0] - sel[k - 1][0]) + abs(sel[k][1] - sel[k - 1][1]) for k in range(1, len(sel))]
            sc += [np.mean(d) if d else 0, np.mean([v > 6 for v in d]) if d else 0]
        else:
            sc += [0] * 9
    resp = [abs(x - pts[i - 1][1]) + abs(y - pts[i - 1][2])
            for i, x, y in mine if i > 0 and pts[i - 1][0] != color]
    sc += [np.mean(resp), np.mean([v <= 3 for v in resp]), np.mean([v >= 10 for v in resp])] if resp else [0, 0, 0]
    sc.append(line(mine[0][1], mine[0][2]))
    return np.concatenate([np.array(sc, dtype=np.float32), op, allm])

# ---- 蒐集候選玩家（每人至少 105 局）----
need = N_GALLERY + 5
pool = []
for p in sorted(glob.glob('training/train_*.csv')):
    d = pd.read_csv(p, usecols=['player_id', 'color', 'sgf_content'])
    c = d.player_id.value_counts()
    ok = c[c >= need].index.tolist()
    rng.shuffle(ok)
    ok = ok[:60]                                     # 每個等級取 60 人 -> 共 ~600 人
    sub = d[d.player_id.isin(ok)]
    for pid, g in sub.groupby('player_id'):
        g = g.sample(n=min(len(g), N_GALLERY + N_QUERY_MAX), random_state=1)
        pool.append(g)
    print(p, 'players picked', len(ok), flush=True)
pool = pd.concat(pool, ignore_index=True)

players = pool.player_id.unique().tolist()
rng.shuffle(players)
players = players[:N_CAND]
pool = pool[pool.player_id.isin(players)].reset_index(drop=True)
print('候選庫人數 %d，總局數 %d' % (len(players), len(pool)))

vecs = defaultdict(list)
for pid, col, sgf in zip(pool.player_id, pool.color, pool.sgf_content):
    v = game_vec(sgf, col)
    if v is not None: vecs[pid].append(v)
print('有效玩家 %d' % len(vecs))

G, Q, names = [], [], []
for pid, vs in vecs.items():
    if len(vs) < 25: continue
    vs = np.stack(vs); rng.shuffle(vs)
    ng = min(N_GALLERY, len(vs) - 5)
    nq = min(N_QUERY_MAX, len(vs) - ng)
    G.append(vs[:ng].mean(0)); Q.append(vs[ng:ng + nq].mean(0)); names.append(pid)
G = np.stack(G); Q = np.stack(Q)
print('gallery %s  query %s' % (G.shape, Q.shape))

mu, sd = G.mean(0), G.std(0) + 1e-8
Gn = (G - mu) / sd; Qn = (Q - mu) / sd
Gn /= np.linalg.norm(Gn, axis=1, keepdims=True); Qn /= np.linalg.norm(Qn, axis=1, keepdims=True)
S = Qn @ Gn.T
order = np.argsort(-S, axis=1)
true = np.arange(len(names))
rank = np.array([np.where(order[i] == true[i])[0][0] for i in range(len(names))])
score = np.where(rank < 5, np.exp(-rank), 0.0)
print()
print('=== 手工特徵地板（%d 人候選庫）===' % len(names))
print('Top-1 accuracy : %.4f' % (rank == 0).mean())
print('Top-5 accuracy : %.4f' % (rank < 5).mean())
print('官方公式分數    : %.4f' % score.mean())
print('中位名次        : %.0f / %d' % (np.median(rank) + 1, len(names)))
