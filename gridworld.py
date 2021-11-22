import pickle

import game
import logging
import networkx as nx
import numpy as np
import os


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
                state_list.append((r, c))
                st_id = r * self.cols + c
                self.graph.add_node(st_id)
        
        if self.conn == 4:
            # Represent (-1, 0), (1, 0), (0, -1), (0, 1)
            act_index = [0, 1, 2, 3]
            self.act = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            # Represent (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)
            act_index = [0, 1, 2, 3, 4, 5, 6, 7]
            self.act = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for st in state_list:
            for a_id in act_index:
                next_st = tuple(np.array(st) + np.array(self.act[a_id]))
                if next_st in state_list:
                    st_id = st[0] * self.cols + st[1]
                    next_st_id = next_st[0] * self.cols + next_st[1]
                    self.graph.add_edge(st_id, next_st_id, action=a_id)
                    
        logger.info("Successfully Get NetworkX Graph")
        

if __name__ == '__main__':

    for i in range(2, 20):
        game_bdd = Gridworld(rows=i, cols=i)
        game_nx = GridworldNx(rows=i, cols=i)
        print(f"gw size: {i, i}, BDD size: {game_bdd.size()}, NX size: {game_nx.size()}")
