import itertools
import pickle

import game
import logging
import networkx as nx
import numpy as np
import os
import util
import time
from guppy import hpy
import matplotlib.pyplot as plt
import guppy
from dd import BDD
import math

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


class Gridworld(game.Game):
    def __init__(self, rows, cols, conn=4):
        assert conn in [4, 8]
        self.rows = rows
        self.cols = cols
        self.conn = conn
        super(Gridworld, self).__init__(num_states=rows * cols, num_actions=conn)

        self._add_gw_states()
        self._add_gw_actions()
        self._add_gw_trans()

    def size(self):
        obj_state = [self.bdd_states, self.bdd_actions, self.bdd_trans]
        with open("tmp.pkl", "wb") as file:
            pickle.dump(obj_state, file)

        size = os.path.getsize("tmp.pkl")
        os.remove("tmp.pkl")
        return size

    def _add_gw_states(self):
        for r in range(self.rows):
            for c in range(self.cols):
                uid = self._cell2uid(r, c)
                self.add_state(uid)

    def _add_gw_actions(self):
        for a in range(self.conn):
            self.add_action(a)

    def _add_gw_trans(self):
        for r in range(self.rows):
            for c in range(self.cols):
                for a in range(self.conn):
                    # TODO. Add transitions
                    pass

    def _uid2cell(self, uid):
        return divmod(uid, self.cols)

    def _cell2uid(self, r, c):
        return r * self.cols + c


class GridworldNx:
    """
    Networkx implementation of gridworld.
    """
    def __init__(self, rows, cols, conn=4):
        assert conn in [4, 8]
        self.rows = rows
        self.cols = cols
        self.conn = conn
        self.gw_dim = (self.rows, self.cols)
        self.graph = nx.MultiDiGraph()
        self._construct_gridworld()

    def size(self):
        obj_state = [self.graph]
        with open("tmp.pkl", "wb") as file:
            pickle.dump(obj_state, file)

        size = os.path.getsize("tmp.pkl")
        os.remove("tmp.pkl")
        return size

    def _construct_gridworld(self):
        # Add states
        # Add edges based on connectivity
        state_list = []
        for r in range(self.rows):
            for c in range(self.cols):
                for p in range(2):
                    st_id = util.cell2uid(r, c, p, self.gw_dim)
                    self.graph.add_node(st_id)
        
        if self.conn == 4:
            # Represent (-1, 0), (1, 0), (0, -1), (0, 1)
            self.act_index = [0, 1, 2, 3]
            self.act = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            # Represent (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)
            self.act_index = [0, 1, 2, 3, 4, 5, 6, 7]
            self.act = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        stlist = list(self.graph.nodes)
        for st in stlist:
            r, c, p = util.uid2cell(st, self.gw_dim)
            for a_id in self.act_index:
                gw_st = (r, c)
                next_st = tuple(np.array(gw_st) + np.array(self.act[a_id]))
                if self.in_Gridworld(next_st):
                    p_ = (p+1)%2
                    next_st_id = util.cell2uid(next_st[0], next_st[1], p_, self.gw_dim)
                    self.graph.add_edge(st_id, next_st_id, action=a_id)
                    
        logger.info("Successfully Get NetworkX Graph")

    def in_Gridworld(self, gw_st):
        if gw_st[0] <= 0 or gw_st[0] >=self.rows or gw_st[1] <= 0 or gw_st[1] >= self.cols:
            return False
        return True

    def preExist(self, st):
        pre = [pred for pred in self.graph.predecessors(st)]
        result = []
        for pred in pre:
            edge_dict = self.graph[pred][st]
            for i in edge_dict.keys():
                act = self.graph[pred][st][i]
                if act in self.act_index:    #Modify here to distinguish between P1 and P2
                    result.append(pred)
                    break
        result_set = set(result)
        return result_set

    def preAll(self, st):
        pre = [pred for pred in self.graph.predecessors(st)]
        result_set = set([])
        for pred in pre:
            edge_dict = self.graph[pred][st]
            actset = set([])
            for i in edge_dict.keys():
                act = self.graph[pred][st][i]
                if act in self.act_index:  #Modify here to distinguish between P1 and P2
                    actset.add(act)
            if actset == set(self.act_index):
                result_set.add(pred)
        return result_set


def Analysis_nx(i):
    h = hpy()
    start_mem = h.heap().size
    start_time = time.time()
    
    game_nx = GridworldNx(rows = i, cols = i)

    end_time = time.time()
    end_mem = h.heap().size
#    print("--- %s seconds ---" % (nx_time))
    
#    print(h.heap)
    runtime_ms = round((end_time - start_time) * 1e6, ndigits=4)
    runtime_mem = end_mem - start_mem
#    runtime_mem = 0
    return runtime_ms, runtime_mem, game_nx

def gridworld_nx_profile(i):
    runtime_ms, runtime_mem, nx_gridworld = Analysis_nx(i)
    
    h = hpy()
    
    end_mem1 = h.heap().size()
    ...   #Fill in how we construct the game using networkx
    end_time2 = time.time()
    end_mem2 = h.heap().size
    runtime2_ms = round((end_time2 - end_time1) * 1e3, ndigits=4)
    runmem2_bytes = end_mem2 - end_mem1
    
    # Inference graph object
    ... #Fill in how we construct igraph using networkx
    end_time3 = time.time()
    end_mem3 = h.heap().size
    runtime3_ms = round((end_time3 - end_time2) * 1e3, ndigits=4)
    runmem3_bytes = end_mem3 - end_mem2

    # Product computation
    ... #Fill in how we construct hypergame
    end_time4 = time.time()
    end_mem4 = h.heap().size
    runtime4_ms = round((end_time4 - end_time3) * 1e3, ndigits=4)
    runmem4_bytes = end_mem4 - end_mem3

    # Compute sure winning states of hypergame
    ... #Fill in the sure winning region in networkx gridworlf
    sw.solve()
    end_time5 = time.time()
    end_mem5 = h.heap().size
    runtime5_ms = round((end_time5 - end_time4) * 1e3, ndigits=4)
    runmem5_bytes = end_mem5 - end_mem4

    # Compute deceptive almost-sure winning states of hypergame
    
    dasw = ...   #Fill in the DASW function for gridworld.
    end_time6 = time.time()
    end_mem6 = h.heap().size
    runtime6_ms = round((end_time6 - end_time5) * 1e3, ndigits=4)
    runmem6_bytes = end_mem6 - end_mem5
    
    timeconsuming = [runtime1_ms, runtime2_ms, runtime3_ms, runtime4_ms, runtime5_ms, runtime6_ms]
    spaceconsuming = [runmem1_bytes, runmem2_bytes, runmem3_bytes, runmem4_bytes, runmem5_bytes, runmem6_bytes]
    

def main_nx():
    time_nx_list = []
    mem_nx_list = []
    for i in range(2, 21):
        nx_time, nx_mem = Analysis_nx(i)
        print(i, nx_time, nx_mem)
        time_nx_list.append(nx_time)
        mem_nx_list.append(nx_mem)
    print(mem_nx_list)
    return time_nx_list


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

    runtime_us = round((end_time - start_time) * 1e6, ndigits=4)
    runtime_mem = end_mem - start_mem

    print(f"count(states): {bdd.count(states)}, count(trans): {bdd.count(trans)}")
    return runtime_us, runtime_mem


if __name__ == '__main__':
    # time_nx_list = main_nx()
    #
    # nrows, ncols = (5, 5)
    # nactions = 4
    # time_bdd = []
    # mem_bdd = []
    # for dim in range(2, 21):
    #     time_ms, mem_bytes = gridworld_bdd_profile(nrows=dim, ncols=dim, nactions=nactions)
    #     print(dim, time_ms, mem_bytes)
    #     time_bdd.append(time_ms)
    #     mem_bdd.append(mem_bytes)
    #
    # print(mem_bdd)
    # print(time_bdd)
    # print(time_nx_list)

    time_bdd = [31520.8435, 33964.8724, 52021.9803, 102000.4749, 180000.5436, 237999.6777, 233875.5131, 391831.6364, 475086.689, 560718.5364, 755488.6341, 902945.9953, 1071426.3916, 1184113.0257, 1186952.8294, 1749562.9787, 2001583.3378, 2160105.7053, 2436623.8117]
    time_nx = [0.0, 0.0, 0.0, 2000.5703, 9003.6392, 11968.1358, 10999.918, 14027.1187, 13999.7005, 13988.4949, 17018.0798, 17998.6954, 19170.5227, 21030.9029, 23966.3124, 26030.0636, 27160.4061, 28635.025, 33996.8204]
    mem_bdd = [152413, 390500, 379370, 1428338, 2657458, 3195705, 2197712, 6316873, 7298030, 8263547, 13181456, 14649926, 17649199, 22773829, 13044388, 28544302, 30977404, 35756449, 46079999]
    mem_nx = [6220, 15748, 32956, 58292, 87092, 128354, 170772, 217783, 283372, 346652, 412712, 484824, 591200, 685464, 776824, 874488, 978456, 1144048, 1279064]

    plt.figure(0)
    plt.scatter(list(range(2, 21)), time_bdd)
    plt.scatter(list(range(2, 21)), time_nx)

    plt.figure(1)
    plt.scatter(list(range(2, 21)), mem_bdd)
    plt.scatter(list(range(2, 21)), mem_nx)
    plt.show()
