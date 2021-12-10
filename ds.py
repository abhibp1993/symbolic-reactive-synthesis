import logging
import math
from abc import abstractmethod
import util

try:
    from dd.cudd import BDD
    print(f"{__name__}.py: Using dd.CUDD package")
except ImportError:
    from dd.bdd import BDD
    print(f"{__name__}.py: Using dd python package")

from dd import BDD

logger = logging.getLogger(__name__)


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
    ID_GEN = None

    def __init__(self, name, **kwargs):
        if self.__class__.ID_GEN is None:
            self.__class__.ID_GEN = IdGen()
        self.id = self.__class__.ID_GEN.next()
        self.name = name
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"S{self.id}"

    def __str__(self):
        return f"State({self.name})"

    @abstractmethod
    def to_binary(self, n, k):
        """
        In case of concurrent game, k=0. For turn-based game, k=1 or 2 depending on whether game is stochastic or not.

        :param n: number of bits to represent state id
        :param k: number of bits to represent player id
        :return: (list) binary representation [pb_{k-1}... pb_0, b_{n-1}, ..., b_0]
        """
        pass


class Action:
    ID_GEN = None

    def __init__(self, name, **kwargs):
        if self.__class__.ID_GEN is None:
            self.__class__.ID_GEN = IdGen()
        self.id = self.__class__.ID_GEN.next()
        self.name = name
        self.__dict__.update(kwargs)

    @abstractmethod
    def to_binary(self, n, k):
        """
        In case of concurrent game, k=0. For turn-based game, k=1 or 2 depending on whether game is stochastic or not.

        :param n: number of bits to represent state id
        :param k: number of bits to represent player id
        :return: (list) binary representation [pb_{k-1}... pb_0, b_{n-1}, ..., b_0]
        """
        pass


class GraphBDD:
    def __init__(self, bdd: BDD):
        self.bdd = bdd

        self.bdd_states_n = 0           # Num(bits) to represent state id
        self.bdd_states_k = 0           # Num(bits) to represent state player
        self.bdd_actions_n = 0          # Num(bits) to represent action id
        self.bdd_actions_k = 0          # Num(bits) to represent action player

        self.bddv_state = None          # bit-varname for state
        self.bddv_state2 = None         # bit-varname for state (primed)
        self.bddv_action = None         # bit-varname for action (primed)
        self.bddv_state_vars = []
        self.bddv_action_vars = []
        self.bddv_trans_vars = []

        self.bddf_state = None          # state validity formula
        self.bddf_action = None         # action validity formula
        self.bddf_trans = None          # trans validity formula

        # Book keeping
        self.num_states = 0
        self.num_actions = 0
        self.num_trans = 0

    def declare(self, max_states, max_actions, var_state, var_state_prime, var_action, nbits_state_p, nbits_action_p):
        # Compute bits needed
        self.bdd_states_n = math.ceil(math.log2(max_states))
        self.bdd_states_k = nbits_state_p
        self.bdd_actions_n = math.ceil(math.log2(max_actions))
        self.bdd_actions_k = nbits_action_p

        # Configure variables, generate variable names
        self.bddv_state = var_state
        self.bddv_state2 = var_state_prime
        self.bddv_action = var_action

        self.bddv_state_vars = [f"p{self.bddv_state}{i}" for i in range(self.bdd_states_k)] + \
                               [f"{self.bddv_state}{i}" for i in range(self.bdd_states_n)]
        self.bddv_action_vars = [f"p{self.bddv_action}{i}" for i in range(self.bdd_actions_k)] + \
                                [f"{self.bddv_action}{i}" for i in range(self.bdd_actions_n)]
        self.bddv_trans_vars = self.bddv_state_vars + self.bddv_action_vars + \
                               [f"p{self.bddv_state2}{i}" for i in range(self.bdd_states_k)] + \
                               [f"{self.bddv_state2}{i}" for i in range(self.bdd_states_n)]

        # Declare/add variables to bdd
        self.bdd.declare(*self.bddv_trans_vars)

    def add_state(self, u: State):
        # Validate ID
        assert u.id < 2 ** self.bdd_states_n

        # Construct formula for state
        u_binary = u.to_binary(n=self.bdd_states_n, k=self.bdd_states_k)
        u_f_str = util.bin2expr(u_binary, self.bddv_state_vars)

        # Update state formula
        u_formula = self.bdd.add_expr(u_f_str)
        self.bddf_state = u_formula if self.bddf_state is None else self.bddf_state | u_formula

        # Update state count
        self.num_states += 1

    def add_action(self, a: Action):
        # Validate ID
        assert a.id < 2 ** self.bdd_actions_n

        # Construct formula for action
        a_binary = a.to_binary(n=self.bdd_actions_n, k=self.bdd_actions_k)
        a_f_str = util.bin2expr(a_binary, self.bddv_action_vars)

        # Update action formula
        a_formula = self.bdd.add_expr(a_f_str)
        self.bddf_action = a_formula if self.bddf_action is None else self.bddf_action | a_formula

        # Update action count
        self.num_actions += 1

    def add_trans(self, u: State, a: Action, v: State):
        assert self.has_state(u)
        assert self.has_state(v)
        assert self.has_action(a)

        # Construct formula string
        u_binary = u.to_binary(n=self.bdd_states_n, k=self.bdd_states_k)
        v_binary = v.to_binary(n=self.bdd_states_n, k=self.bdd_states_k)
        a_binary = a.to_binary(n=self.bdd_actions_n, k=self.bdd_actions_k)

        u_f_str = util.bin2expr(u_binary, self.bddv_state_vars)
        v_f_str = util.bin2expr(v_binary, self.bddv_state_vars).replace(self.bddv_state, self.bddv_state2)
        a_f_str = util.bin2expr(a_binary, self.bddv_action_vars)

        f_str = " & ".join([u_f_str, a_f_str, v_f_str])

        # Update transition validity formula
        formula = self.bdd.add_expr(f_str)
        self.bddf_trans = formula if self.bddf_trans is None else self.bddf_trans | formula

        # Increment count of transitions
        self.num_trans += 1

    def has_state(self, u: State):
        # Validate ID
        if u.id >= 2 ** self.bdd_states_n:
            return False

        #
        u_binary = u.to_binary(n=self.bdd_states_n, k=self.bdd_states_k)
        u_dict = dict(zip(self.bddv_state_vars, [(True if bit == 1 else False) for bit in u_binary]))
        out = self.bdd.let(u_dict, self.bddf_state)
        if out == self.bdd.true:
            return True
        elif out == self.bdd.false:
            return False
        else:
            print(f"self.bdd.let(...) -> {out.to_expr()}. Did not evaluate to true/false.")
            raise ValueError(f"self.bdd.let(...) -> {out.to_expr()}. Did not evaluate to true/false.")

    def has_action(self, a: Action):
        if a.id >= 2 ** self.bdd_actions_n:
            return False
        a_binary = a.to_binary(n=self.bdd_actions_n, k=self.bdd_actions_k)
        a_dict = dict(zip(self.bddv_action_vars, [(True if bit == 1 else False) for bit in a_binary]))
        out = self.bdd.let(a_dict, self.bddf_action)
        if out == self.bdd.true:
            return True
        elif out == self.bdd.false:
            return False
        else:
            raise ValueError(f"self.bdd.let(...) -> {out.to_expr()}. Did not evaluate to true/false.")

    def has_trans(self, u: State, a: Action, v: State):
        if not self.has_state(u) or not self.has_state(v) or not self.has_action(a):
            return False

        # Build substitution dictionary
        u_binary = u.to_binary(n=self.bdd_states_n, k=self.bdd_states_k)
        v_binary = v.to_binary(n=self.bdd_states_n, k=self.bdd_states_k)
        a_binary = a.to_binary(n=self.bdd_actions_n, k=self.bdd_actions_k)

        u_dict = dict(zip(self.bddv_state_vars, [(True if bit == 1 else False) for bit in u_binary]))
        v_dict = dict(zip([var.replace(self.bddv_state, self.bddv_state2) for var in self.bddv_state_vars],
                          [(True if bit == 1 else False) for bit in v_binary]))
        a_dict = dict(zip(self.bddv_action_vars, [(True if bit == 1 else False) for bit in a_binary]))
        subs_dict = u_dict | a_dict | v_dict

        # Evaluate transition validity formula
        out = self.bdd.let(subs_dict, self.bddf_trans)
        if out == self.bdd.true:
            return True
        elif out == self.bdd.false:
            return False
        else:
            raise ValueError(f"self.bdd.let(...) -> {out.to_expr()}. Did not evaluate to true/false.")


class GameBDD(GraphBDD):
    def __init__(self, bdd):
        super(GameBDD, self).__init__(bdd)
        self.bddf_final = None

    def make_final(self, u):
        # Validate ID
        assert u.id < 2 ** self.bdd_states_n

        # Construct formula for state
        u_binary = u.to_binary(n=self.bdd_states_n, k=self.bdd_states_k)
        u_f_str = util.bin2expr(u_binary, self.bddv_state_vars)

        # Update state formula
        u_formula = self.bdd.add_expr(u_f_str)
        self.bddf_final = u_formula if self.bddf_final is None else self.bddf_final | u_formula

        # Update state count
        self.num_states += 1


class HypergameBDD:
    def __init__(self, bdd: BDD):
        self.bdd = bdd

        self.bdd_states_n = 0       # Num(bits) to represent state id
        self.bdd_states_k = 0       # Num(bits) to represent state player
        self.bdd_actions_n = 0      # Num(bits) to represent action id
        self.bdd_actions_k = 0      # Num(bits) to represent action player

        self.bddv_state = None      # bit-varname for state
        self.bddv_state2 = None     # bit-varname for state (primed)
        self.bddv_action = None     # bit-varname for action (primed)
        self.bddv_state_vars = []
        self.bddv_action_vars = []
        self.bddv_trans_vars = []

        self.bddf_state = None  # state validity formula
        self.bddf_action = None  # action validity formula
        self.bddf_trans = None  # trans validity formula
        self.bddf_final = None

        # Book keeping
        self.num_states = 0
        self.num_actions = 0
        self.num_trans = 0

    def declare(self, game: GameBDD, igraph: GraphBDD):
        # State validity function
        self.bddf_state = game.bddf_state & igraph.bddf_state

        # Action validity
        self.bddf_action = game.bddf_action

        # Transition validity
        self.bddf_trans = game.bddf_trans & igraph.bddf_trans & self.bdd.add_expr("pa0 <-> pb0")

    def has_state(self, s, q):
        pass

    def has_action(self, a):
        pass

    def has_trans(self, s, p, a, t, q):
        pass


class SW:
    def __init__(self, hg: HypergameBDD):
        self.hg = hg
        self.bdd = hg.bdd
        self.p1_win_states = self.bdd.false
    
    def least_point_computation(self):
        # Get final states
        final = self.hg.bddf_final

        # Initialize loop variables
        z = self.bdd.false                              # Represents states currently labeled as winning for P1
        y = None                                        # Intermediate loop variable

        # prime variables
        prime_subs = dict()                             # Construct the corresponding relations between u and v
        v_vars = set()                                  # The element should be the v variables
        a1_vars = set()                                 # The element should be the action variable for P1
        a2_vars = set()                                 # The element should be the action variable for P2

        pa0 = self.bdd.add_expr("pa0")
        pu0 = self.bdd.add_expr("pu0")
        pb0 = self.bdd.add_expr("pb0")
        pv0 = self.bdd.add_expr("pv0")

        while z != y:
            # Update loop variable
            y = z

            # Rename variables in z to make them target
            #   u_{} -> v_{}, i_{} -> j_{}
            next_q = self.bdd.let(prime_subs, z)

            # Pre1: {s \in S1 | \exists a \in A1: T(s, a) \in Y}
            u1 = self.hg.bddf_trans & next_q & pa0 & pu0
            pre1 = self.bdd.quantify(v_vars, u1, forall=False)

            # Pre2: {s \in S2 | \forall a \in A2: T(s, a) \in Y}
            u2 = self.hg.bddf_trans & next_q & pb0 & pv0
            pre2 = self.bdd.quantify(v_vars, u2, forall=False)

            # Quantify pre1, pre2 to represent states
            #   Since pre1: S x A1 -> {T, F}, pre2: S x A2 -> {T, F}
            pre1 = self.bdd.quantify(a1_vars, pre1, forall=False)
            pre2 = self.bdd.quantify(a2_vars, pre2, forall=True)

            # Update winning states
            #   Now, pre1: S -> {T, F} and pre2: S -> {T, F}
            z = z | pre1 | pre2 | final

        return z


class DASW:
    def __init__(self, bdd):
        pass


def product(bdd: BDD, game: GameBDD, igraph: GraphBDD):
    print(bdd.vars)
    hg = HypergameBDD(bdd)
    hg.declare(game, igraph)
    return hg