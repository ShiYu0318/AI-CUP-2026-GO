"""手工特徵 + GBDT 的棋力預測地板實驗。
目的：(1) 測出不用深度學習能到多少分  (2) 實證「期望分數解碼」相對 argmax 的增益
驗證協定：player-disjoint split（模擬 Private LB 的未見玩家）

⚠️ **2026-09-18 的早期探索腳本，已被取代，目前沒有任何程式引用它。**
結論已進 RESULTS.md：解碼器比較見 §2.3、XGBoost vs MLP 見 §3.2。
現行的 GBDT 實作在 `scripts/train_gbdt_ensemble.py`。
它不依賴 `src/aicup/`，可獨立重跑；但評估協定是當時的版本，
**絕對數字不能與現在的比較**。
"""
import glob, re, sys, numpy as np, pandas as pd
from collections import defaultdict

LEVELS = ['D','C','B','A','1D','2D','3D','4D','5D','6D']
R2I = {r:i for i,r in enumerate(LEVELS)}
N_PER_RANK = int(sys.argv[1]) if len(sys.argv)>1 else 6000
rng = np.random.default_rng(0)
MOVE_RE = re.compile(r';([BW])\[([a-s])([a-s])\]')

def feats(sgf, color):
    mv = MOVE_RE.findall(sgf)
    if len(mv) < 20: return None
    pts = [(c, ord(x)-97, ord(y)-97) for c,x,y in mv]
    n = len(pts)
    mine = [(i,x,y) for i,(c,x,y) in enumerate(pts) if c==color]
    opp  = [(i,x,y) for i,(c,x,y) in enumerate(pts) if c!=color]
    if len(mine) < 10: return None
    f = {'n_moves': n, 'n_mine': len(mine), 'is_black': 1.0 if color=='B' else 0.0}

    def line(x,y): return min(x,y,18-x,18-y)          # 0 = 第一線
    def phase(i):                                      # 開局/中盤/官子
        return 0 if i<40 else (1 if i<120 else 2)

    # 各階段的離邊距離與線分布
    for p in range(3):
        sel = [(x,y) for i,x,y in mine if phase(i)==p]
        f[f'p{p}_cnt'] = len(sel)
        if sel:
            ls = [line(x,y) for x,y in sel]
            f[f'p{p}_line_mean'] = float(np.mean(ls))
            f[f'p{p}_line_std']  = float(np.std(ls))
            for L in range(5):
                f[f'p{p}_line{L}'] = float(np.mean([l==L for l in ls]))
        else:
            f[f'p{p}_line_mean']=f[f'p{p}_line_std']=np.nan
            for L in range(5): f[f'p{p}_line{L}']=np.nan

    # 連續己手的跳躍距離（tenuki 傾向）
    for p in range(3):
        sel = [(x,y) for i,x,y in mine if phase(i)==p]
        d = [abs(sel[k][0]-sel[k-1][0])+abs(sel[k][1]-sel[k-1][1]) for k in range(1,len(sel))]
        f[f'p{p}_jump_mean'] = float(np.mean(d)) if d else np.nan
        f[f'p{p}_tenuki']    = float(np.mean([v>6 for v in d])) if d else np.nan

    # 應手率：己手與「前一手對方落子」的距離
    prev = {}
    for i,(c,x,y) in enumerate(pts): prev[i]=(x,y)
    resp = [abs(x-prev[i-1][0])+abs(y-prev[i-1][1]) for i,x,y in mine if i-1 in prev and pts[i-1][0]!=color]
    if resp:
        f['resp_mean'] = float(np.mean(resp))
        f['resp_local'] = float(np.mean([v<=3 for v in resp]))   # 貼身應手比例
        f['resp_far']   = float(np.mean([v>=10 for v in resp]))  # 脫先比例
    else:
        f['resp_mean']=f['resp_local']=f['resp_far']=np.nan

    # 首手與開局佔角
    f['first_line'] = float(line(mine[0][1], mine[0][2]))
    corners = set()
    for i,x,y in mine[:6]:
        corners.add((x<9, y<9))
    f['corners_first6'] = len(corners)

    # 己方落子的空間離散度
    xs = np.array([x for _,x,_ in mine]); ys = np.array([y for _,_,y in mine])
    f['spread_x'] = float(xs.std()); f['spread_y'] = float(ys.std())
    f['center_frac'] = float(np.mean([(abs(x-9)<=4 and abs(y-9)<=4) for _,x,y in mine]))

    # 官子階段第一/二線比重（收官技巧的代理）
    endg = [(x,y) for i,x,y in mine if i>=int(0.75*n)]
    f['endgame_line01'] = float(np.mean([line(x,y)<=1 for x,y in endg])) if endg else np.nan
    return f

rows, meta = [], []
for lv in LEVELS:
    d = pd.read_csv(f'training/train_{lv}.csv', usecols=['player_id','rank','color','sgf_content'])
    d = d.sample(n=min(N_PER_RANK,len(d)), random_state=0)
    for pid, col, sgf in zip(d.player_id, d.color, d.sgf_content):
        f = feats(sgf, col)
        if f is None: continue
        rows.append(f); meta.append((pid, R2I[lv]))
    print(f'{lv} done, cum={len(rows)}', flush=True)

X = pd.DataFrame(rows)
meta = pd.DataFrame(meta, columns=['player_id','y'])
print('X', X.shape)

# ---- player-disjoint split ----
players = meta.groupby('player_id')['y'].first()
hold = []
for r, g in players.groupby(players):
    ids = g.index.to_numpy(); rng.shuffle(ids)
    hold += list(ids[:max(1, int(len(ids)*0.25))])
hold = set(hold)
m = meta.player_id.isin(hold).to_numpy()
print('train games %d / holdout games %d / holdout players %d' % ((~m).sum(), m.sum(), len(hold)))

from sklearn.ensemble import HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
                                     l2_regularization=1.0, random_state=0)
clf.fit(X[~m], meta.y[~m])
P = clf.predict_proba(X[m])
yv = meta.y[m].to_numpy(); pv = meta.player_id[m].to_numpy()
print('單局 top-1 acc: %.4f' % (P.argmax(1)==yv).mean())

# ---- 題目層級模擬：每題 5~20 局同一玩家 ----
def comp_score(pred, true):
    return np.where(pred==true, 1.0, np.where(np.abs(pred-true)==1, np.exp(-1), 0.0))

E = np.zeros((10,10))                       # E[k,j] = 預測 k、真值 j 的得分
for k in range(10):
    for j in range(10):
        E[k,j] = 1.0 if k==j else (np.exp(-1) if abs(k-j)==1 else 0.0)

by = defaultdict(list)
for i,p in enumerate(pv): by[p].append(i)
qs = []
for p, idx in by.items():
    if len(idx) < 5: continue
    for _ in range(3):
        k = rng.integers(5, min(20, len(idx))+1)
        qs.append((rng.choice(idx, k, replace=False), yv[idx[0]]))
print('模擬題數', len(qs))

a_arg, a_exp = [], []
for idx, t in qs:
    p = P[idx].mean(0)
    a_arg.append(comp_score(np.array(p.argmax()), t))
    a_exp.append(comp_score(np.array((E @ p).argmax()), t))
print('題目層級  argmax 解碼      : %.4f' % np.mean(a_arg))
print('題目層級  期望分數解碼      : %.4f' % np.mean(a_exp))
print('期望分數解碼增益           : %+.4f' % (np.mean(a_exp)-np.mean(a_arg)))
imp = pd.Series(clf.feature_importances_ if hasattr(clf,'feature_importances_') else np.zeros(X.shape[1]), index=X.columns)
