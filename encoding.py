import logging
import util

logger = logging.getLogger(__name__)

try:
    from dd.cudd import BDD
    logger.info("Using dd.CUDD package")
except ImportError:
    from dd import BDD
    logger.info("Using dd python package")

logging.basicConfig(level=logging.DEBUG)


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
        self.id = self.__class__.ID_GEN.next()
        self.name = name
        self.turn = turn

    def __repr__(self):
        return f"S{self.id}"

    def __str__(self):
        return f"State({self.name})"

    def to_binary(self):
        pass

    def from_binary(self, bin):
        pass


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

    def init_bdd_vars(self, bdd_vars_states, bdd_vars_actions, u_varname="u", v_varname="v"):
        self.bdd_vars_states = bdd_vars_states + [bitname.replace(u_varname, v_varname) for bitname in bdd_vars_states]
        self.bdd_vars_actions = bdd_vars_actions
        self.bdd.declare(*self.bdd_vars_states, *self.bdd_vars_actions)

    def add_state(self, state: State, varname="u"):
        # Get state id
        uid = state.id
        player = state.turn
        assert uid < 2**(len(self.bdd_vars_states)/2)

        # Get binary expression corresponding to uid
        uid_expr = util.id2expr(id=uid, num_bits=len(self.bdd_vars_states) // 2 - 1, varname=varname)
        uid_expr = uid_expr + " & " + (f"{varname}p" if player == 1 else f"~{varname}p")
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

    def add_trans(self, u: State, a: Action, v: State, u_varname="u", v_varname="v"):
        # Get ids
        uid, up = u.id, u.turn
        aid, ap = a.id, a.player
        vid, vp = v.id, v.turn

        assert uid < 2 ** (len(self.bdd_vars_states) // 2)
        assert vid < 2 ** (len(self.bdd_vars_states) // 2)
        assert aid < 2 ** len(self.bdd_vars_actions)

        # Get binary expression corresponding to uid, aid and vid
        uid_expr = util.id2expr(id=uid, num_bits=len(self.bdd_vars_states) // 2 - 1, varname=u_varname)
        uid_expr = uid_expr + " & " + (f"{u_varname}p" if up == 1 else f"~{u_varname}p")
        aid_expr = util.id2expr(id=aid, num_bits=len(self.bdd_vars_actions) - 1, varname="a")
        aid_expr = aid_expr + " & " + ("ap" if ap == 1 else "~ap")
        vid_expr = util.id2expr(id=vid, num_bits=len(self.bdd_vars_states) // 2 - 1, varname=v_varname)
        vid_expr = vid_expr + " & " + (f"{v_varname}p" if vp == 1 else f"~{v_varname}p")
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
    game_bdd_vars_states = game.bdd_vars_states
    game_bdd_vars_actions = game.bdd_vars_actions
    igraph_bdd_vars_states = igraph.bdd_vars_states
    igraph_bdd_vars_actions = igraph.bdd_vars_actions
    assert len(set.intersection(set(game_bdd_vars_states), set(igraph_bdd_vars_states))) == 0
    assert game_bdd_vars_actions == igraph_bdd_vars_actions
    # TODO: Variable names need to be handled.

    hg_bdd_vars_states = game_bdd_vars_states + igraph_bdd_vars_states
    hg_bdd_vars_actions = game_bdd_vars_actions
    hypergame.init_bdd_vars(bdd_vars_states=hg_bdd_vars_states, bdd_vars_actions=hg_bdd_vars_actions)

    # Add states
    hg_bddf_state = game.bddf_states & igraph.bddf_states
    hypergame.bddf_states = hg_bddf_state

    # Add transitions
    hg_bddf_trans = game.bddf_trans & igraph.bddf_trans


def main():
    class GameState(State):
        ID_GEN = IdGen()

    # Game: toy problem
    game = GameBDD()
    game.init_bdd_vars(bdd_vars_states=['u0', 'u1', 'up'], bdd_vars_actions=['a0', 'a1', 'ap'])

    states = [GameState(name='s0', turn=2), GameState(name='s1', turn=1), GameState(name='s2', turn=2),
              GameState(name='s3', turn=1)]
    actions = [Action(name='a1', player=1), Action(name='a2', player=1), Action(name='b1', player=2),
               Action(name='b2', player=2)]

    game.add_state(states[0])
    game.add_state(states[1])
    game.add_state(states[2])
    game.add_state(states[3])

    game.add_action(actions[0])
    game.add_action(actions[1])
    game.add_action(actions[2])
    game.add_action(actions[3])

    game.add_trans(states[0], actions[2], states[0])
    game.add_trans(states[0], actions[3], states[0])
    game.add_trans(states[1], actions[0], states[0])
    game.add_trans(states[1], actions[1], states[2])
    game.add_trans(states[2], actions[2], states[1])
    game.add_trans(states[2], actions[3], states[3])
    game.add_trans(states[3], actions[0], states[2])
    game.add_trans(states[3], actions[1], states[2])

    game.make_final(states[0])

    # IGraph
    graph = GraphBDD()
    graph.init_bdd_vars(['i0', 'ip'], ['a0', 'a1', 'ap'], u_varname="i", v_varname="j")

    class GState(State):
        ID_GEN = IdGen()

    igraph_states = [GState(name="i0", turn=1), GState(name='i1', turn=1)]
    igraph_actions = [actions[0], actions[1]]

    graph.add_state(igraph_states[0], varname="i")
    graph.add_state(igraph_states[1], varname="i")

    graph.add_action(igraph_actions[0])
    graph.add_action(igraph_actions[1])

    graph.add_trans(igraph_states[0], igraph_actions[0], igraph_states[1], u_varname="i", v_varname="j")
    graph.add_trans(igraph_states[0], igraph_actions[1], igraph_states[0], u_varname="i", v_varname="j")
    graph.add_trans(igraph_states[1], igraph_actions[0], igraph_states[1], u_varname="i", v_varname="j")
    graph.add_trans(igraph_states[1], igraph_actions[1], igraph_states[1], u_varname="i", v_varname="j")


    hg = product(game, graph)


if __name__ == '__main__':
    main()
    # u0 = State(name="u0", turn=1)
    # u1 = State(name="u1", turn=1)
    # u2 = State(name="u2", turn=2)
    #
    # g = GraphBDD()
    # g.init_bdd_vars(bdd_vars_states=["u0", "u1", "up"], bdd_vars_actions=["a0", "ap"])
    # g.add_state(u0)
    # g.add_state(u1)
    # # g.add_state(u2)
    # print("g.has_state(u0): ", g.has_state(u0))
    # print("g.has_state(u1): ", g.has_state(u1))
    # print("g.has_state(u2): ", g.has_state(u2))
