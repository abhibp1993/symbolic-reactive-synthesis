"""
Profiling.
"""
import itertools

from dd import BDD
import math
import util
import time
import guppy


def gridworld_bdd_profile(nrows, ncols, nactions):
    h = guppy.hpy()
    start_mem = h.heap().size

    start_time = time.time()
    bdd = BDD()

    bits_states = math.ceil(math.log2(nrows * ncols))
    bits_actions = math.ceil(math.log2(nactions))

    bdd_state_vars_u = [f'u{i}' for i in range(bits_states)] + ["p"]
    bdd_state_vars_v = [f'v{i}' for i in range(bits_states)] + ["p"]
    bdd_action_vars = [f'a{i}' for i in range(bits_actions)] + ["p"]

    bdd.declare(*bdd_state_vars_u)
    bdd.declare(*bdd_state_vars_v)
    bdd.declare(*bdd_action_vars)

    # Add states
    states = None
    for i in range(nrows * ncols * 2):
        if states is None:
            states = bdd.add_expr(util.id2expr(i, bits_states, 'u'))
        else:
            states = states | bdd.add_expr(f"({util.id2expr(i, bits_states, 'u')})")

    # Add transitions
    trans = None
    for r, c, p in itertools.product(range(nrows), range(ncols), range(2)):
        uid = util.cell2uid(r, c, p, (nrows, ncols))

        # Action: N
        aid = 1
        nr, nc = r, min(c + 1, ncols)
        np = (p + 1) % 2
        vid = util.cell2uid(nr, nc, np, (nrows, ncols))
        expr_n = " & ".join([util.id2expr(uid, bits_states, "u"),
                             util.id2expr(aid, bits_actions, "a"),
                             util.id2expr(vid, bits_states, "v")])

        # Action: E
        aid = 1
        nr, nc = min(r + 1, nrows), c
        np = (p + 1) % 2
        vid = util.cell2uid(nr, nc, np, (nrows, ncols))
        expr_e = " & ".join([util.id2expr(uid, bits_states, "u"),
                             util.id2expr(aid, bits_actions, "a"),
                             util.id2expr(vid, bits_states, "v")])

        # Action: S
        aid = 1
        nr, nc = r, max(c - 1, 0)
        np = (p + 1) % 2
        vid = util.cell2uid(nr, nc, np, (nrows, ncols))
        expr_s = " & ".join([util.id2expr(uid, bits_states, "u"),
                             util.id2expr(aid, bits_actions, "a"),
                             util.id2expr(vid, bits_states, "v")])

        # Action: W
        aid = 1
        nr, nc = max(r - 1, 0), c
        np = (p + 1) % 2
        vid = util.cell2uid(nr, nc, np, (nrows, ncols))
        expr_w = " & ".join([util.id2expr(uid, bits_states, "u"),
                             util.id2expr(aid, bits_actions, "a"),
                             util.id2expr(vid, bits_states, "v")])

        if trans is None:
            trans = bdd.add_expr(f"({expr_n}) | ({expr_e}) | ({expr_s}) | ({expr_w})")
        else:
            trans = trans | bdd.add_expr(f"({expr_n}) | ({expr_e}) | ({expr_s}) | ({expr_w})")

    end_time = time.time()
    end_mem = h.heap().size

    runtime_ms = round((end_time - start_time) * 1e3, ndigits=4)
    runtime_mem = end_mem - start_mem
    return runtime_ms, runtime_mem


if __name__ == '__main__':
    nrows, ncols = (5, 5)
    nactions = 4
    time_bdd = []
    for dim in range(2, 21):
        time_ms, mem_bytes = gridworld_bdd_profile(nrows=dim, ncols=dim, nactions=nactions)
        print(dim, time_ms, mem_bytes)
        time_bdd.append(time_ms)

    print(time_bdd)
