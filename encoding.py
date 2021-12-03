try:
    from dd.cudd import BDD
except ImportError:
    from dd import BDD


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

    def __init__(self, name, *args, **kwargs):
        self.id = State.ID_GEN.next()
        self.name = name

    def __repr__(self):
        return f"S{self.id}"

    def __str__(self):
        return f"State({self.name})"


class Action:
    ID_GEN = IdGen()

    def __init__(self, name, *args, **kwargs):
        self.id = Action.ID_GEN.next()
        self.name = name


class GraphBDD:
    def __init__(self):
        # BDD representation
        self.bdd = BDD()
        self.bddf_states = None
        self.bddf_actions = None
        self.bddf_trans = None

        # State/Action dict (id: state/action object)
        self.states = dict()
        self.actions = dict()

    def init_bdd_vars(self, bit_names):
        self.bdd.declare(bit_names)

    def add_state(self, state, player=None):
        pass

    def add_action(self, action, player=None):
        pass


class GameBDD(GraphBDD):
    def __init__(self):
        super(GameBDD, self).__init__()
        self.bddf_final = None

    def make_final(self, state):
        pass


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
    u0 = State(name="u0")
    u1 = State(name="u1")
    u2 = State(name="u2")
    print(repr(u0), repr(u1), repr(u2))
    print(str(u0), str(u1), str(u2))
