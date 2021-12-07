try:
    from dd.cudd import BDD
except ImportError:
    from dd import BDD
import itertools
import logging
import math
import util


class IdGen:
    def __init__(self):
        self._curr_id = -1
        self._unused_id = list()

    def next(self):
        if len(self._unused_id) > 0:
            return self._unused_id.pop()
        self._curr_id += 1
        return self._curr_id

    def free(self, uid):
        if uid == self._curr_id:
            self._curr_id -= 1
        else:
            self._unused_id.append(uid)

    def max_id(self):
        return self._curr_id

    def num_active_ids(self):
        return self._curr_id + 1 - len(self._unused_id)


class State:
    ID_GEN = IdGen()

    def __init__(self, name, turn, *args, **kwargs):
        self.id = State.ID_GEN.next()
        self.name = name
        self.turn = turn

    def __repr__(self):
        return f"S{self.id}"

    def __str__(self):
        return f"State({self.name})"


class Action:
    ID_GEN = IdGen()

    def __init__(self, name, player, *args, **kwargs):
        self.id = Action.ID_GEN.next()
        self.name = name
        self.player = player


class GraphBDD:
    def __init__(self):
        # BDD representation
        self.bdd = BDD()
        self.bdd_vars_states = list()
        self.bdd_vars_actions = list()
        self.bddf_states = None
        self.bddf_actions = None
        self.bddf_trans = None

        # State/Action dict (id: state/action object)
        self.states = dict()
        self.actions = dict()

        # Book keeping
        self.num_states = 0
        self.num_actions = 0
        self.num_trans = 0

    def init_bdd_vars(self, bdd_vars_states, bdd_vars_actions):
        self.bdd_vars_states = bdd_vars_states
        self.bdd_vars_actions = bdd_vars_actions
        self.bdd.declare(*bdd_vars_states, *bdd_vars_actions)

    def add_state(self, state: State):
        # Get state id
        uid = state.id
        player = state.turn
        assert uid < 2**len(self.bdd_vars_states)

        # Get binary expression corresponding to uid
        uid_expr = util.id2expr(id=uid, num_bits=len(self.bdd_vars_states) - 1, varname="u")
        uid_expr = uid_expr + " & " + "up" if player == 1 else "~up"
        f = self.bdd.add_expr(uid_expr)

        # Update state validity expression
        self.bddf_states = f if self.bddf_states is None else self.bddf_states | f

        # Save state in object list
        self.states[uid] = state

        # Increment number of active states
        self.num_states += 1

    def add_action(self, action: Action):
        # Get state id
        aid = action.id
        player = action.player
        assert aid < 2 ** len(self.bdd_vars_actions)

        # Get binary expression corresponding to uid
        aid_expr = util.id2expr(id=aid, num_bits=len(self.bdd_vars_actions) - 1, varname="a")
        if player == 1:
            aid_expr = aid_expr + " & " + "ap"
        elif player == 2:
            aid_expr = aid_expr + " & " + "~ap"
        else:
            aid_expr = aid_expr + " & " + "(ap | ~ap)"      # for completeness. Can be removed.

        f = self.bdd.add_expr(aid_expr)

        # Update action validity expression
        self.bddf_actions = f if self.bddf_actions is None else self.bddf_actions | f

        # Save state in object list
        self.actions[aid] = action

        # Increment number of active states
        self.num_actions += 1

    def add_trans(self, u: State, a: Action, v: State):
        # Get ids
        uid, up = u.id, u.turn
        aid, ap = a.id, a.player
        vid, vp = v.id, v.turn

        assert uid < 2 ** len(self.bdd_vars_states)
        assert vid < 2 ** len(self.bdd_vars_states)
        assert aid < 2 ** len(self.bdd_vars_actions)

        # Get binary expression corresponding to uid, aid and vid
        uid_expr = util.id2expr(id=uid, num_bits=len(self.bdd_vars_states), varname="u")
        aid_expr = util.id2expr(id=aid, num_bits=len(self.bdd_vars_actions), varname="a")
        vid_expr = util.id2expr(id=vid, num_bits=len(self.bdd_vars_states), varname="v")
        f = self.bdd.add_expr(" & ".join([uid_expr, aid_expr, vid_expr]))

        # Update transition validity expression
        self.bddf_trans = f if self.bddf_trans is None else self.bddf_trans | f

        # Increment number of active transitions
        self.num_trans += 1

    def has_state(self, u):
        if isinstance(u, int):
            return u in self.states.keys()

        elif isinstance(u, State):
            uid = u.id
            return uid in self.states.keys()

    def has_action(self, a):
        if isinstance(a, int):
            return a in self.actions.keys()

        elif isinstance(a, State):
            uid = a.id
            return uid in self.actions.keys()

    def has_trans(self, u: State, a: Action, v: State):
        pass


class GameBDD(GraphBDD):
    def __init__(self):
        super(GameBDD, self).__init__()
        self.bddf_final = None

    def make_final(self, state):
        # Get state id
        uid = state.id
        player = state.turn
        assert uid < 2 ** len(self.bdd_vars_states)

        # Get binary expression corresponding to uid
        uid_expr = util.id2expr(id=uid, num_bits=len(self.bdd_vars_states), varname="u")
        uid_expr = uid_expr + " & " + "up" if player == 1 else "~up"
        f = self.bdd.add_expr(uid_expr)

        # Update state validity expression
        self.bddf_final = f if self.bddf_final is None else self.bddf_final | f


def product(game: GameBDD, igraph: GraphBDD):
    # Define hypergame BDD
    hypergame = GameBDD()

    # Construct hypergame BDD bit names from game and inference graph names
    state_bit_names = []
    action_bit_names = []
    bit_names = []
    hypergame.init_bdd_vars(bit_names)

    # Add states
    # TODO. The following will need renaming of variables.
    # hg_bddf_state = game.bddf_states & igraph.bddf_states

    # Add transitions
    # TODO. Think how to represent this.
    pass


if __name__ == '__main__':
    u0 = State(name="u0", turn=1)
    u1 = State(name="u1", turn=1)
    u2 = State(name="u2", turn=2)

    g = GraphBDD()
    g.init_bdd_vars(bdd_vars_states=["u0", "u1", "up"], bdd_vars_actions=["a0", "ap"])
    g.add_state(u0)
    g.add_state(u1)
    # g.add_state(u2)
    print("g.has_state(u0): ", g.has_state(u0))
    print("g.has_state(u1): ", g.has_state(u1))
    print("g.has_state(u2): ", g.has_state(u2))
