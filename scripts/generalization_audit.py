"""泛化性稽核：檢驗解法文件中幾條「可能只在訓練分布成立」的假設。

E1  gallery/query 用 game_id 順序切分 vs 隨機切分
    → 若順序切分明顯掉分，代表資料有時序漂移，靠「開局定式偏好」辨識會不穩
E2  盤面位置特徵「8 重對稱折疊」vs「不折疊」
    → 若不折疊明顯較好，代表落子方位帶身分訊號，
      那麼對子題一套用 dihedral augmentation 會破壞訊號
E3  各棋力等級的辨識難度差異
    → 若差異大，真實候選庫的等級組成會直接左右分數，模擬基準需涵蓋多種組成
E4  特徵族消融：純統計量 / 只有開局分布 / 只有全局分布
    → 看分數集中在哪一族，評估風險集中度

⚠️ **2026-09-18 的早期探索腳本，目前沒有任何程式引用它——但不要刪。**
它是幾個**現在還在生效**的決定的唯一可重跑證據，最重要的是 E2：
`src/aicup/dataset.py` 今天仍在用的 8 重對稱增強（`_dihedral`）就是這裡定案的。
E3（各等級辨識難度差異）見 RESULTS.md §1.3，E1/E4 見 §1.4。
它不依賴 `src/aicup/`，可獨立重跑；但評估協定是當時的版本，
**絕對數字不能與現在的比較**。
"""
import glob, re, numpy as np, pandas as pd
from collections import defaultdict

MOVE_RE = re.compile(r';([BW])\[([a-s])([a-s])\]')
rng = np.random.default_rng(7)
N_CAND, N_GALLERY, N_QUERY = 400, 100, 20

FOLD, _c = {}, []
for x in range(19):
    for y in range(19):
        a, b = min(x, 18 - x), min(y, 18 - y)
        k = (min(a, b), max(a, b))
        if k not in FOLD:
            FOLD[k] = len(_c); _c.append(k)
NF = len(_c)                                   # 55
# 不折疊：19x19 降解析度到 10x10 以控制維度
def cell_raw(x, y): return (x // 2) * 10 + (y // 2)
NR = 100

def parse(sgf, color):
    mv = MOVE_RE.findall(sgf)
    if len(mv) < 30: return None
    pts = [(c, ord(x) - 97, ord(y) - 97) for c, x, y in mv]
    mine = [(i, x, y) for i, (c, x, y) in enumerate(pts) if c == color]
    if len(mine) < 15: return None
    def fold(x, y):
        a, b = min(x, 18 - x), min(y, 18 - y)
        return FOLD[(min(a, b), max(a, b))]
    def line(x, y): return min(x, y, 18 - x, 18 - y)

    op_f, all_f = np.zeros(NF), np.zeros(NF)
    op_r, all_r = np.zeros(NR), np.zeros(NR)
    for k, (i, x, y) in enumerate(mine):
        all_f[fold(x, y)] += 1; all_r[cell_raw(x, y)] += 1
        if k < 15: op_f[fold(x, y)] += 1; op_r[cell_raw(x, y)] += 1
    for a in (op_f, all_f, op_r, all_r): a /= max(a.sum(), 1)

    sc = [len(pts), len(mine), 1.0 if color == 'B' else 0.0]
    for lo, hi in [(0, 40), (40, 120), (120, 10**9)]:
        sel = [(x, y) for i, x, y in mine if lo <= i < hi]
        if sel:
            ls = [line(x, y) for x, y in sel]
            sc += [np.mean(ls), np.std(ls)] + [np.mean([l == L for l in ls]) for L in range(5)]
            d = [abs(sel[k][0]-sel[k-1][0])+abs(sel[k][1]-sel[k-1][1]) for k in range(1, len(sel))]
            sc += [np.mean(d) if d else 0, np.mean([v > 6 for v in d]) if d else 0]
        else: sc += [0]*9
    resp = [abs(x-pts[i-1][1])+abs(y-pts[i-1][2]) for i, x, y in mine if i > 0 and pts[i-1][0] != color]
    sc += [np.mean(resp), np.mean([v <= 3 for v in resp]), np.mean([v >= 10 for v in resp])] if resp else [0,0,0]
    sc.append(line(mine[0][1], mine[0][2]))
    return np.array(sc, np.float32), op_f, all_f, op_r, all_r

# ---- 蒐集：保留 game_id 順序 ----
store = {}                                      # pid -> (rank, [ (gid, parts) ... ] 依 game_id 排序)
for p in sorted(glob.glob('training/train_*.csv')):
    rk = p.split('train_')[1][:-4]
    d = pd.read_csv(p, usecols=['player_id','game_id','color','sgf_content'])
    cnt = d.player_id.value_counts()
    ok = cnt[cnt >= N_GALLERY + N_QUERY].index.tolist()
    rng.shuffle(ok); ok = ok[:60]
    sub = d[d.player_id.isin(ok)].copy()
    sub['gnum'] = sub.game_id.str.extract(r'(\d+)').astype(int)
    for pid, g in sub.groupby('player_id'):
        g = g.sort_values('gnum')
        rows = []
        for gid, col, sgf in zip(g.gnum, g.color, g.sgf_content):
            r = parse(sgf, col)
            if r is not None: rows.append((gid, r))
        if len(rows) >= N_GALLERY + 5: store[pid] = (rk, rows)
    print(rk, 'players kept', sum(1 for v in store.values() if v[0] == rk), flush=True)

pids = list(store); rng.shuffle(pids); pids = pids[:N_CAND]
print('候選庫 %d 人' % len(pids))

def assemble(rows, idx_g, idx_q, which):
    """which: 'all' | 'scalar' | 'open' | 'global' | 'raw'（raw = 不折疊位置）"""
    def vec(r):
        sc, op_f, all_f, op_r, all_r = r
        if which == 'all':    return np.concatenate([sc, op_f, all_f])
        if which == 'scalar': return sc
        if which == 'open':   return op_f
        if which == 'global': return all_f
        if which == 'raw':    return np.concatenate([sc, op_r, all_r])
    G = np.stack([vec(rows[i][1]) for i in idx_g]).mean(0)
    Q = np.stack([vec(rows[i][1]) for i in idx_q]).mean(0)
    return G, Q

def evaluate(split, which, nq=N_QUERY):
    G, Q, rk = [], [], []
    for pid in pids:
        r, rows = store[pid]
        n = len(rows)
        if split == 'random':
            o = rng.permutation(n)
        else:                                   # 'ordered'：依 game_id 前後切
            o = np.arange(n)
        ng = min(N_GALLERY, n - nq)
        g, q = assemble(rows, o[:ng], o[ng:ng+nq], which)
        G.append(g); Q.append(q); rk.append(r)
    G, Q = np.stack(G), np.stack(Q)
    mu, sd = G.mean(0), G.std(0) + 1e-8
    Gn, Qn = (G-mu)/sd, (Q-mu)/sd
    Gn /= np.linalg.norm(Gn,axis=1,keepdims=True); Qn /= np.linalg.norm(Qn,axis=1,keepdims=True)
    o = np.argsort(-(Qn @ Gn.T), 1)
    r = np.array([np.where(o[i]==i)[0][0] for i in range(len(G))])
    return np.where(r<5, np.exp(-r), 0).mean(), (r==0).mean(), np.array(rk), r

print('\n=== E1：gallery/query 切分方式（特徵 = all, query 20 局）===')
for sp in ['random','ordered']:
    s,t1,_,_ = evaluate(sp,'all')
    print('  %-8s  分數 %.4f   Top-1 %.4f' % (sp, s, t1))

print('\n=== E2：位置特徵是否做 8 重對稱折疊 ===')
for w,label in [('all','折疊(55格)'),('raw','不折疊(10x10)')]:
    s,t1,_,_ = evaluate('random',w)
    print('  %-14s  分數 %.4f   Top-1 %.4f' % (label, s, t1))

print('\n=== E3：各等級的辨識難度（特徵 = all, random split）===')
s,t1,rk,r = evaluate('random','all')
df = pd.DataFrame({'rank':rk,'score':np.where(r<5,np.exp(-r),0),'top1':(r==0)})
g = df.groupby('rank').agg(n=('score','size'), score=('score','mean'), top1=('top1','mean'))
order=['D','C','B','A','1D','2D','3D','4D','5D','6D']
print(g.reindex([x for x in order if x in g.index]).to_string(float_format=lambda v:'%.4f'%v))

print('\n=== E4：特徵族消融（random split, query 20 局）===')
for w,label in [('scalar','僅統計量(42維)'),('open','僅開局分布(55維)'),
                ('global','僅全局分布(55維)'),('all','全部(152維)')]:
    s,t1,_,_ = evaluate('random',w)
    print('  %-18s  分數 %.4f   Top-1 %.4f' % (label, s, t1))
