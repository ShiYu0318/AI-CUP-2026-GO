"""走法模型的局面與目標點必須對齊。

錯一手（取到「我方落子之後」的局面）不會報錯——模型只會學到「預測一個已經有子的點」，
被遮罩逼成亂猜；左右翻轉只翻盤面不翻目標點也一樣。這些測試把對齊釘死。
"""
import numpy as np
import torch

from aicup.goboard import parse_moves
from aicup.skillpolicy import (EMPTY_PLANE, N_LEVELS, SkillPolicyNet, game_loglik, own_turns,
                               position_samples)

# 80 個互不重複的點（7 與 361 互質），確保是一盤合法、不會下在有子之處的棋。
SGF = "(;SZ[19]" + "".join(
    f";{'B' if i % 2 == 0 else 'W'}[{chr(97 + (i * 7) % 361 % 19)}{chr(97 + (i * 7) % 361 // 19)}]"
    for i in range(80)) + ")"


def test_own_turns_are_ours_and_skip_the_first_move():
    for color, parity in (("B", 0), ("W", 1)):
        t, tg = own_turns(SGF, color)
        assert t and all(x % 2 == parity and x > 0 for x in t)
        mv = parse_moves(SGF)
        assert tg == [mv[x][1] * 19 + mv[x][2] for x in t]


def test_target_point_is_empty_in_the_sampled_position():
    """局面是「前一手下完之後」，所以我方要下的點此刻必須是空的。"""
    t, tg = own_turns(SGF, "W")
    planes, tgt = position_samples(SGF, "W", t, tg)
    assert planes.shape[0] == len(t)
    empty = planes[:, EMPTY_PLANE].reshape(len(t), -1)
    assert all(empty[i, tgt[i]] == 1 for i in range(len(t)))


def test_horizontal_flip_keeps_target_aligned():
    t, tg = own_turns(SGF, "B")
    planes, tgt = position_samples(SGF, "B", t, tg)
    fp = planes[..., ::-1]
    ft = (tgt // 19) * 19 + (18 - tgt % 19)
    empty = fp[:, EMPTY_PLANE].reshape(len(t), -1)
    assert all(empty[i, ft[i]] == 1 for i in range(len(t)))


def test_occupied_points_get_zero_probability_and_rows_normalise():
    net = SkillPolicyNet(width=16, blocks=2).eval()
    t, tg = own_turns(SGF, "B")
    planes, _ = position_samples(SGF, "B", t[:3], tg[:3])
    from aicup.planes import planes_to_float
    x = torch.from_numpy(planes_to_float(planes))
    with torch.no_grad():
        lp = net(x)
    assert lp.shape == (3, N_LEVELS, 361)
    p = lp.exp()
    assert torch.allclose(p.sum(-1), torch.ones(3, N_LEVELS), atol=1e-4)
    occ = (x[:, EMPTY_PLANE] < 0.5).flatten(1)
    assert float(p[occ[:, None, :].expand_as(p)].max()) < 1e-6


def test_game_loglik_shape_and_move_counts():
    net = SkillPolicyNet(width=16, blocks=2)
    G = game_loglik(net, [(SGF, "B"), (SGF, "W")], "cpu", max_moves=10)
    assert G.shape == (2, 2 + N_LEVELS)
    assert (G[:, 0] <= 10).all() and (G[:, 1] >= G[:, 0]).all()
    assert np.isfinite(G[:, 2:]).all() and (G[:, 2:] < 0).all()
