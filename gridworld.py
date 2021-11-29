import pickle

#import game
import logging
import networkx as nx
import numpy as np
import os
import util
import time
from guppy import hpy

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

        stlist = list(G.nodes)
        for st in stlist:
            r, c, p = util.uid2cell(st, self.gw_dim)
            for a_id in self.act_index:
                gw_st = (r, c)
                next_st = tuple(np.array(gw_st) + np.array(self.act[a_id]))
                if self.in_Gridworld(next_st):
                    p_ = (p+1)%2
                    next_st_id = util.cell2uid(next_st[0], next_st[1], p_)
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

def Analysis_nx(rows, cols):
    start_time = time.time()
    game_nx = GridworldNx(rows = i, cols = i)
    nx_time = time.time()- start_time
    print("--- %s seconds ---" % (nx_time))
    h = hpy()
    print(h.heap)

    return game_nx
if __name__ == '__main__':

    for i in range(2, 20):
#s        game_bdd = Gridworld(rows=i, cols=i)
        game_nx = Analysis_nx(rows=i, cols=i)
#        print(f"gw size: {i, i}, BDD size: {game_bdd.size()}, NX size: {game_nx.size()}")
