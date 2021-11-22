import networkx as nx

import game
import logging
import numpy as np
logger = logging.getLogger(__name__)


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
        pass

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
        pass

    def _construct_gridworld(self):
        # Add states
        # Add edges based on connectivity
        statelist = []
        for i in range(self.rows):
            for j in range(self.cols):
                statelist.append((i, j))
                st_id = i * self.rows + j
                self.graph.add_node(st_id)
        
        if self.conn == 4:
            self.actindex = [0, 1, 2, 3]  #Represent (-1, 0), (1, 0), (0, -1), (0, 1)
            self.act = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            self.actindex = [0, 1, 2, 3, 4, 5, 6, 7] #Represent (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)
            self.act = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        
        for st in statelist:
            for a_id in self.actindex:
                next_st = tuple(np.array(st) + np.array(self.act[a_id]))
                if next_st in statelist:
                    st_id = st[0] * self.rows + st[1]
                    next_st_id = next_st[0] * self.rows + next_st[1]
                    self.graph.add_edge(st_id, next_st_id, action = a_id)
                    
        print("Successfully Get NexworkX Graph")
        
    


if __name__ == '__main__':
    game = Gridworld(rows=4, cols=4)
    
    game_ = GridworldNx(rows = 4, cols = 4)
