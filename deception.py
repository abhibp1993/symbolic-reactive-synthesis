"""
Almost-sure winning in deterministic two-player turn-based game with action misperception.
A. Kulkarni and J. Fu, IJCAI'20
"""

from dd import BDD
import itertools
import logging
import math
import util

logger = logging.getLogger(__name__)


class GraphBDD:
    def __init__(self, max_states, max_actions):
        # Size of graph (fixed: cannot be changed during runtime)
        self.max_states = max_states
        self.max_actions = max_actions

        # Define BDD
        self.bdd = BDD()
        self.num_states = 0
        self.num_actions = 0
        self.num_trans = 0

        # Initialize state and action bit names
        self.bits_states = math.ceil(math.log2(max_states))
        self.bits_actions = math.ceil(math.log2(max_actions))
        self.bdd_state_vars = [f'u{i}' for i in range(self.bits_states)]
        self.bdd_state_vars2 = [f'v{i}' for i in range(self.bits_states)]
        self.bdd_action_vars = [f'a{i}' for i in range(self.bits_actions)]
        self.bdd.declare(*self.bdd_state_vars)
        self.bdd.declare(*self.bdd_action_vars)
        self.bdd.declare(*self.bdd_state_vars2)

        # BDD functions
        self.bddf_p1_states = None
        self.bddf_p2_states = None
        self.bddf_p1_actions = None
        self.bddf_p2_actions = None
        self.bddf_trans = None

        # State and transition properties
        self.state_props = dict()
        self.action_props = dict()
        self.edge_props = dict()

    def add_state(self, uid, player, udict=None):
        assert uid < self.max_states

        # Get binary expression corresponding to uid
        uid_expr = util.id2expr(id=uid, num_bits=self.bits_states, varname="u")
        f = self.bdd.add_expr(uid_expr)

        # Update state validity expression
        if player == 1:
            self.bddf_p1_states = f if self.bddf_p1_states is None else self.bddf_p1_states | f
        elif player == 2:
            self.bddf_p2_states = f if self.bddf_p2_states is None else self.bddf_p2_states | f
        else:
            logger.error(f"Player must be 1 or 2.")

        # Update state properties
        self.state_props[uid] = udict

        # Increment number of active states
        self.num_states += 1

    def add_action(self, aid, player, adict=None):
        assert aid < self.max_actions

        # Get binary expression corresponding to uid
        aid_expr = util.id2expr(id=aid, num_bits=self.bits_actions, varname="a")
        f = self.bdd.add_expr(aid_expr)

        # Update state validity expression
        if player == 1:
            self.bddf_p2_actions = f if self.bddf_p2_actions is None else self.bddf_p2_actions | f
        elif player == 2:
            self.bddf_p2_actions = f if self.bddf_p2_actions is None else self.bddf_p2_actions | f
        else:
            logger.error(f"Player must be 1 or 2.")

        # Update state properties
        self.action_props[aid] = adict

        # Increment number of active actions
        self.num_actions += 1

    def add_trans(self, uid, aid, vid, edict=None):
        # TODO. check if uid and aid belong to same player.
        assert self.has_state(uid) and self.has_state(vid) and self.has_action(aid)

        # Get binary expression corresponding to uid, aid and vid
        uid_expr = util.id2expr(id=uid, num_bits=self.bits_states, varname="u")
        aid_expr = util.id2expr(id=aid, num_bits=self.bits_actions, varname="a")
        vid_expr = util.id2expr(id=vid, num_bits=self.bits_states, varname="v")
        f = self.bdd.add_expr(" & ".join([uid_expr, aid_expr, vid_expr]))

        print(" & ".join([uid_expr, aid_expr, vid_expr]))
        print(f.to_expr())

        # Update transition validity expression
        self.bddf_trans = f if self.bddf_trans is None else self.bddf_trans | f

        # Update edge properties
        self.edge_props[(uid, aid, vid)] = edict

        # Increment number of active transitions
        self.num_trans += 1

    def has_state(self, uid, player=None):
        # ID cannot be larger than maximum number of states in graph
        if uid >= self.max_states:
            return False

        # Get dictionary for uid
        uid_dict = util.id2dict(uid, self.bits_states, varname="u")

        # Evaluate expression
        if player is None:
            val = self.bdd.let(uid_dict, self.bddf_p1_states | self.bddf_p2_states)
        elif player == 1:
            val = self.bdd.let(uid_dict, self.bddf_p1_states)
        elif player == 2:
            val = self.bdd.let(uid_dict, self.bddf_p2_states)
        else:
            raise ValueError("Unknown player")

        # Interpret evaluated expression
        if val.to_expr() == 'FALSE':
            return False
        elif val.to_expr() == 'TRUE':
            return True
        else:
            raise ValueError("Unknown value")

    def has_action(self, aid, player=None):
        # ID cannot be larger than maximum number of actions in graph
        if aid >= self.max_actions:
            return False

        # Get dictionary for uid
        aid_dict = util.id2dict(aid, self.bits_actions, varname="a")

        # Evaluate expression
        if player is None:
            val = self.bdd.let(aid_dict, self.bddf_p2_actions | self.bddf_p2_actions)
        elif player == 1:
            val = self.bdd.let(aid_dict, self.bddf_p2_actions)
        elif player == 2:
            val = self.bdd.let(aid_dict, self.bddf_p2_actions)
        else:
            raise ValueError("Unknown player")

        # Interpret evaluated expression
        if val.to_expr() == 'FALSE':
            return False
        elif val.to_expr() == 'TRUE':
            return True
        else:
            raise ValueError("Unknown value")

    def has_trans(self, uid, aid, vid):
        # ID cannot be larger than maximum number of actions in graph
        if uid >= self.max_states or aid >= self.max_actions or vid >= self.max_states:
            return False

        # Get dictionary for uid
        uid_dict = util.id2dict(uid, self.bits_states, varname="u")
        aid_dict = util.id2dict(aid, self.bits_actions, varname="a")
        vid_dict = util.id2dict(vid, self.bits_states, varname="v")
        dict_ = uid_dict | aid_dict | vid_dict

        # Evaluate expression
        val = self.bdd.let(dict_, self.bddf_trans)

        # Interpret evaluated expression
        if val.to_expr() == 'FALSE':
            return False
        elif val.to_expr() == 'TRUE':
            return True
        else:
            raise ValueError("Unknown value")

    def pred(self, vid, aid=None):
        # ID's cannot be larger than maximum states/actions in graph
        assert vid < self.max_states
        if aid is not None:
            assert aid < self.max_actions

        # Get dictionary for vid, aid
        aid_dict = util.id2dict(aid, self.bits_actions, varname="a") if aid is not None else dict()
        vid_dict = util.id2dict(vid, self.bits_states, varname="v")
        dict_ = aid_dict | vid_dict

        # Substitute known vid, aid values in transition formula
        f = self.bdd.let(dict_, self.bddf_trans)
        pred = self.bdd.quantify(f, self.bdd_action_vars, forall=False)

        return pred

    def succ(self, uid, aid=None):
        # ID's cannot be larger than maximum states/actions in graph
        assert uid < self.max_states
        if aid is not None:
            assert aid < self.max_actions

        # Get dictionary for vid, aid
        aid_dict = util.id2dict(aid, self.bits_actions, varname="a") if aid is not None else dict()
        uid_dict = util.id2dict(uid, self.bits_states, varname="u")
        dict_ = aid_dict | uid_dict

        # Substitute known vid, aid values in transition formula
        f = self.bdd.let(dict_, self.bddf_trans)
        succ = self.bdd.quantify(f, self.bdd_action_vars, forall=False)

        return succ


class GameBDD(GraphBDD):
    def __init__(self, max_states, max_actions):
        super(GameBDD, self).__init__(max_states, max_actions)
        self.bddf_final_states = None

    def make_state_final(self, uid):
        assert self.has_state(uid)

        # Get binary expression corresponding to uid
        uid_expr = util.id2expr(id=uid, num_bits=self.bits_states, varname="u")
        f = self.bdd.add_expr(uid_expr)

        # Update final state validity function
        self.bddf_final_states = f if self.bddf_final_states is None else self.bddf_final_states | f


class Gridworld(GameBDD):
    def __init__(self, nrows, ncols, actions):
        """ actions: list of {"name": "", "act": ()} """
        self.nrows = nrows
        self.ncols = ncols
        self.actions = {i: actions[i] for i in range(len(actions))}
        super(Gridworld, self).__init__(max_states=nrows * ncols, max_actions=len(actions))

        self._add_gw_states()
        self._add_gw_actions()
        self._add_gw_trans()

    def _add_gw_states(self):
        for r, c, p in itertools.product(range(self.nrows), range(self.ncols), range(2)):
            uid = self._cell2uid(r, c)
            self.add_state(uid, player=p, udict={"name": f"({r}, {c})"})

    def _add_gw_actions(self):
        for i in range(len(self.actions)):
            self.add_action(i, player=1, adict={"name": self.actions[i]["name"]})
            self.add_action(i, player=2, adict={"name": self.actions[i]["name"]})

    def _add_gw_trans(self):
        pass

    def _uid2cell(self, uid):
        return divmod(uid, self.ncols)

    def _cell2uid(self, r, c):
        return r * self.ncols + c


def product(game: GameBDD, igraph: GraphBDD):
    # Create a hypergame
    hg = GameBDD(game.max_states * igraph.max_states, game.max_actions)

    # Variable substitution
    gvar_subs = {v: v.replace("b", "gb") for v in game.bdd_state_vars}
    ivar_subs = {v: v.replace("b", "ib") for v in igraph.bdd_state_vars}

    # Add states to hypergame
    hg.bddf_p1_states = game.bdd.let(gvar_subs, game.bddf_p1_states) & igraph.bdd.let(ivar_subs, igraph.bddf_p1_states)
    hg.bddf_p2_states = game.bdd.let(gvar_subs, game.bddf_p2_states) & igraph.bdd.let(ivar_subs, igraph.bddf_p2_states)


def demo_igraph():
    # IJCAI extension (arxiv) example. Inference graph.
    igraph = GraphBDD(max_states=4, max_actions=9)

    igraph.add_state(uid=0, player=1)
    igraph.add_state(uid=1, player=1)
    igraph.add_state(uid=2, player=1)
    igraph.add_state(uid=3, player=1)

    igraph.add_action(aid=0, player=1, adict={"name": "N"})
    igraph.add_action(aid=1, player=1, adict={"name": "E"})
    igraph.add_action(aid=2, player=1, adict={"name": "S"})
    igraph.add_action(aid=3, player=1, adict={"name": "W"})
    igraph.add_action(aid=4, player=1, adict={"name": "Cut"})
    igraph.add_action(aid=5, player=1, adict={"name": "JumpN"})
    igraph.add_action(aid=6, player=1, adict={"name": "JumpE"})
    igraph.add_action(aid=7, player=1, adict={"name": "JumpS"})
    igraph.add_action(aid=8, player=1, adict={"name": "JumpW"})

    igraph.add_trans(uid=0, aid=0, vid=0)
    igraph.add_trans(uid=0, aid=1, vid=0)
    igraph.add_trans(uid=0, aid=2, vid=0)
    igraph.add_trans(uid=0, aid=3, vid=0)
    igraph.add_trans(uid=0, aid=4, vid=1)
    igraph.add_trans(uid=0, aid=5, vid=2)
    igraph.add_trans(uid=0, aid=6, vid=2)
    igraph.add_trans(uid=0, aid=7, vid=2)
    igraph.add_trans(uid=0, aid=8, vid=2)

    igraph.add_trans(uid=1, aid=0, vid=1)
    igraph.add_trans(uid=1, aid=1, vid=1)
    igraph.add_trans(uid=1, aid=2, vid=1)
    igraph.add_trans(uid=1, aid=3, vid=1)
    igraph.add_trans(uid=1, aid=4, vid=1)
    igraph.add_trans(uid=1, aid=5, vid=3)
    igraph.add_trans(uid=1, aid=6, vid=3)
    igraph.add_trans(uid=1, aid=7, vid=3)
    igraph.add_trans(uid=1, aid=8, vid=3)

    igraph.add_trans(uid=2, aid=0, vid=2)
    igraph.add_trans(uid=2, aid=1, vid=2)
    igraph.add_trans(uid=2, aid=2, vid=2)
    igraph.add_trans(uid=2, aid=3, vid=2)
    igraph.add_trans(uid=2, aid=4, vid=3)
    igraph.add_trans(uid=2, aid=5, vid=2)
    igraph.add_trans(uid=2, aid=6, vid=2)
    igraph.add_trans(uid=2, aid=7, vid=2)
    igraph.add_trans(uid=2, aid=8, vid=2)

    igraph.add_trans(uid=3, aid=0, vid=3)
    igraph.add_trans(uid=3, aid=1, vid=3)
    igraph.add_trans(uid=3, aid=2, vid=3)
    igraph.add_trans(uid=3, aid=3, vid=3)
    igraph.add_trans(uid=3, aid=4, vid=3)
    igraph.add_trans(uid=3, aid=5, vid=3)
    igraph.add_trans(uid=3, aid=6, vid=3)
    igraph.add_trans(uid=3, aid=7, vid=3)
    igraph.add_trans(uid=3, aid=8, vid=3)

    print("Inference graph constructed.")

    # Test graph
    for i in range(0, 6):
        print(f"state id: {i} is in igraph: {igraph.has_state(i)}")

    for i, j, k in itertools.product(range(4), range(9), range(4)):
        print(f"edge({i, j, k} in igraph: {igraph.has_trans(i, j, k)}")

    return igraph


def demo_gridworld():
    pass


if __name__ == '__main__':
    g = demo_igraph()
