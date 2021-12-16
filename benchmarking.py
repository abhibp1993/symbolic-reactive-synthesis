from ds import *
import guppy
import itertools
import networkx as nx
import random
import time


DIM = [d for d in range(5, 20)]


class GameState(State):
    def __init__(self, name, player):
        assert player == 0 or player == 1
        super(GameState, self).__init__(name)
        self.player = player

    def to_binary(self, k, n):
        """
        Representation: First k-bits are player bits, following n-bits are state id.
            [pb_{k-1}, ..., pb_0, b_{n-1}, ..., b0]
        """
        binary = util.id2bin(self.player, k) + util.id2bin(self.id, n)
        return [int(bit) for bit in binary]

    @staticmethod
    def from_binary(binary, k, n):
        """
        Representation: First k-bits are player bits, following n-bits are state id.
            [pb_{k-1}, ..., pb_0, b_{n-1}, ..., b0]
        """
        assert len(binary) == k + n
        player = util.bin2id(binary[:k])
        sid = util.bin2id(binary[k:])
        return player, sid


class GameAction(Action):
    def __init__(self, name, player):
        assert player == 0 or player == 1
        super(GameAction, self).__init__(name)
        self.player = player

    def to_binary(self, k, n):
        """
        Representation: First k-bits are player bits, following n-bits are state id.
            [pb_{k-1}, ..., pb_0, b_{n-1}, ..., b0]
        """
        binary = util.id2bin(self.player, k) + util.id2bin(self.id, n)
        return [int(bit) for bit in binary]

    @staticmethod
    def from_binary(binary, k, n):
        """
        Representation: First k-bits are player bits, following n-bits are state id.
            [pb_{k-1}, ..., pb_0, b_{n-1}, ..., b0]
        """
        assert len(binary) == k + n
        player = util.bin2id(binary[:k])
        sid = util.bin2id(binary[k:])
        return player, sid


class IGraphState(State):
    def __init__(self, name):
        super(IGraphState, self).__init__(name)

    def to_binary(self, k, n):
        """
        Representation: n-bits are state id.
            [b_{n-1}, ..., b0]
        """
        binary = util.id2bin(self.id, n)
        return [int(bit) for bit in binary]

    @staticmethod
    def from_binary(binary, k, n):
        """
        Representation: First k-bits are player bits, following n-bits are state id.
            [pb_{k-1}, ..., pb_0, b_{n-1}, ..., b0]
        """
        assert len(binary) == k + n
        player = util.bin2id(binary[:k])
        sid = util.bin2id(binary[k:])
        return player, sid


def symbolic_construction(bdd, dim):
    max_states = 2 * dim ** 4    # st: (p1.x, p1.y, p2.x, p2.y, p)
    max_actions = 4 * 2          # {N, E, S, W} for P1 and P2

    game = GameBDD(bdd)
    game.declare(max_states=max_states, max_actions=max_actions, var_action="a", var_state="u", var_state_prime="v",
              nbits_state_p=1, nbits_action_p=1)

    g_actions = [
        GameAction(name="n0", player=0),
        GameAction(name="e0", player=0),
        GameAction(name="s0", player=0),
        GameAction(name="w0", player=0),
        GameAction(name="n1", player=1),
        GameAction(name="e1", player=1),
        GameAction(name="s1", player=1),
        GameAction(name="w1", player=1)
    ]
    for act in g_actions:
        game.add_action(act)

    g_states = []
    for x1, y1, x2, y2, p in itertools.product(range(dim), range(dim), range(dim), range(dim), [0, 1]):
        st = GameState(name=(x1, y1, x2, y2), player=p)
        game.add_state(st)
        g_states.append(st)

    # A dim x dim 4-conn gridworld has 4 * (dim - 1) ^ 2 + 3 * 4 * (dim - 2) + 8 edges.
    # We will add these many edges arbitrarily.
    g_edges = set()
    for i in range(4 * (dim - 1) ^ 2 + 3 * 4 * (dim - 2) + 8):
        u = random.choice(g_states)
        a = random.choice([act for act in g_actions if act.player == u.player])
        v = random.choice(g_states)
        g_edges.add((u, a, v))
        game.add_trans(u, a, v)

    game.make_final(g_states[0])
    game.make_final(g_states[1])

    # Information graph (fixed)
    igraph = GraphBDD(bdd)
    igraph.declare(max_states=8, max_actions=3, var_action="b", var_state="i", var_state_prime="j",
              nbits_state_p=0, nbits_action_p=1)

    g_states = [
        IGraphState(name="{N}}"),
        IGraphState(name="{NE}}"),
        IGraphState(name="{NEW}}"),
        IGraphState(name="{NES}}"),
        IGraphState(name="{NEWS}}")
    ]

    g_actions = [act for act in g_actions if act.player == 0]

    for state in g_states:
        igraph.add_state(state)

    for act in g_actions:
        igraph.add_action(act)

    igraph.add_trans(g_states[0], g_actions[0], g_states[1])
    igraph.add_trans(g_states[0], g_actions[1], g_states[0])
    igraph.add_trans(g_states[1], g_actions[0], g_states[1])
    igraph.add_trans(g_states[1], g_actions[1], g_states[1])

    return game, igraph


def symbolic_solution(game, igraph):
    hypergame = product(game.bdd, game, igraph)
    dasw = DASW(hg=hypergame)
    dasw.solve()
    return dasw


def enumeration_construction(dim):
    game = nx.MultiDiGraph()
    g_actions = [
        GameAction(name="n0", player=0),
        GameAction(name="e0", player=0),
        GameAction(name="s0", player=0),
        GameAction(name="w0", player=0),
        GameAction(name="n1", player=1),
        GameAction(name="e1", player=1),
        GameAction(name="s1", player=1),
        GameAction(name="w1", player=1)
    ]
    game.actions = g_actions

    g_states = []
    for x1, y1, x2, y2, p in itertools.product(range(dim), range(dim), range(dim), range(dim), [0, 1]):
        st = GameState(name=(x1, y1, x2, y2), player=p)
        game.add_node(st)
        g_states.append(st)

    # A dim x dim 4-conn gridworld has 4 * (dim - 1) ^ 2 + 3 * 4 * (dim - 2) + 8 edges.
    # We will add these many edges arbitrarily.
    g_edges = set()
    for i in range(4 * (dim - 1) ^ 2 + 3 * 4 * (dim - 2) + 8):
        u = random.choice(g_states)
        a = random.choice([act for act in g_actions if act.player == u.player])
        v = random.choice(g_states)
        g_edges.add((u, a, v))
        game.add_edge(u, v, action=a)

    game.final = {g_states[0], g_states[1]}

    # Information graph (fixed)
    igraph = nx.MultiDiGraph()

    g_states = [
        IGraphState(name="{N}}"),
        IGraphState(name="{NE}}"),
        IGraphState(name="{NEW}}"),
        IGraphState(name="{NES}}"),
        IGraphState(name="{NEWS}}")
    ]

    g_actions = [act for act in g_actions if act.player == 0]
    igraph.actions = g_actions

    for state in g_states:
        igraph.add_node(state)

    igraph.add_edge(g_states[0], g_states[1], action=g_actions[0])
    igraph.add_edge(g_states[0], g_states[0], action=g_actions[1])
    igraph.add_edge(g_states[1], g_states[1], action=g_actions[0])
    igraph.add_edge(g_states[1], g_states[1], action=g_actions[1])

    return game, igraph


def enumeration_solution(game, igraph):
    hypergame = product_nx(game, igraph)
    dasw = DASW_Nx(hg=hypergame)
    dasw.solve()
    return dasw


def benchmark(dim, niter, type_="symbolic"):
    avg_mem_const = 0
    avg_mem_solve = 0
    avg_time_const = 0
    avg_time_solve = 0

    for icount in range(niter):
        # Reset bdd and other generators
        bdd = BDD()
        GameState.ID_GEN = IdGen()
        IGraphState.ID_GEN = IdGen()
        GameAction.ID_GEN = IdGen()

        # Initialize memory, time measurement
        h = guppy.hpy()
        mem_start = h.heap().size
        time_start = time.time()

        # Construct gridworld
        if type_ == "symbolic":
            game, igraph = symbolic_construction(bdd, dim)
        else:
            game, igraph = enumeration_construction(dim)

        # Measure memory, time
        mem_construction = h.heap().size
        time_construction = time.time()

        # Solve game
        if type_ == "symbolic":
            symbolic_solution(game, igraph)
        else:
            enumeration_solution(game, igraph)

        # Measure memory, time
        mem_solution = h.heap().size
        time_solution = time.time()

        avg_mem_const += mem_construction - mem_start
        avg_mem_solve += mem_solution - mem_construction
        avg_time_const += time_construction - time_start
        avg_time_solve += time_solution - time_construction

        print(icount, avg_mem_const, avg_mem_solve, avg_time_const, avg_time_solve)

    return avg_mem_const / niter, avg_mem_solve / niter, avg_time_const / niter, avg_time_solve / niter


if __name__ == '__main__':
    # benchmark(5, 2, "symbolic")
    benchmark(5, 2, "enumeration")
    print("OK")