"""
Example 1 implements toy problem in https://arxiv.org/pdf/2104.11676.pdf

Modification:
    * Controlled player (P1) is denoted as Player 0
    * Adversarial player (P2) is denoted as Player 1
"""
from ds import *
import guppy
import time
import milestone1


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


def construct_game(bdd):
    g = GameBDD(bdd)
    g.declare(max_states=4, max_actions=4, var_action="a", var_state="u", var_state_prime="v", nbits_state_p=1,
              nbits_action_p=1)

    g_states = []
    for i in range(4):
        p = (i + 1) % 2
        g_states.append(GameState(name=f"s{i}", player=p))

    g_actions = [
        GameAction(name="a1", player=0),
        GameAction(name="a2", player=0),
        GameAction(name="b1", player=1),
        GameAction(name="b2", player=1)
    ]

    for state in g_states:
        g.add_state(state)

    for act in g_actions:
        g.add_action(act)

    g.add_trans(g_states[0], g_actions[2], g_states[0])
    g.add_trans(g_states[0], g_actions[3], g_states[0])
    g.add_trans(g_states[1], g_actions[0], g_states[0])
    g.add_trans(g_states[1], g_actions[1], g_states[2])
    g.add_trans(g_states[2], g_actions[2], g_states[1])
    g.add_trans(g_states[2], g_actions[3], g_states[3])
    g.add_trans(g_states[3], g_actions[0], g_states[2])
    g.add_trans(g_states[3], g_actions[1], g_states[2])

    g.make_final(g_states[0])

    return g, g_states, g_actions


def construct_igraph(bdd, game_actions):
    g = GraphBDD(bdd)
    g.declare(max_states=2, max_actions=2, var_action="b", var_state="i", var_state_prime="j",
              nbits_state_p=0, nbits_action_p=1)

    g_states = [
        IGraphState(name="i0"),
        IGraphState(name="i1")
    ]

    g_actions = [act for act in game_actions if act.player == 0]

    for state in g_states:
        g.add_state(state)

    for act in g_actions:
        g.add_action(act)

    g.add_trans(g_states[0], g_actions[0], g_states[1])
    g.add_trans(g_states[0], g_actions[1], g_states[0])
    g.add_trans(g_states[1], g_actions[0], g_states[1])
    g.add_trans(g_states[1], g_actions[1], g_states[1])

    return g, g_states, g_actions

def time_and_memory_profile(nrows, ncols, nactions):
    runtime1_ms, runmem1_bytes, bdd = milestone1.gridworld_bdd_profile(nrows=nrows, ncols=ncols, nactions=nactions)
    
    end_mem1 = h.heap().size()
    game, game_states, game_actions = construct_game(bdd)   #Replace this with the construct function, if needed
    end_time2 = time.time()
    end_mem2 = h.heap().size
    runtime2_ms = round((end_time2 - end_time1) * 1e3, ndigits=4)
    runmem2_bytes = end_mem2 - end_mem1
    
    # Inference graph object
    igraph, igraph_states, igraph_actions = construct_igraph(bdd, game_actions)  #Replace this with the construct function, if needed
    end_time3 = time.time()
    end_mem3 = h.heap().size
    runtime3_ms = round((end_time3 - end_time2) * 1e3, ndigits=4)
    runmem3_bytes = end_mem3 - end_mem2

    # Product computation
    hypergame = product(bdd, game, igraph)  #Replace this with the construct function, if needed
    end_time4 = time.time()
    end_mem4 = h.heap().size
    runtime4_ms = round((end_time4 - end_time3) * 1e3, ndigits=4)
    runmem4_bytes = end_mem4 - end_mem3

    # Compute sure winning states of hypergame
    sw = SW(hg=hypergame)    #Replace this with the construct function, if needed
    sw.solve()
    end_time5 = time.time()
    end_mem5 = h.heap().size
    runtime5_ms = round((end_time5 - end_time4) * 1e3, ndigits=4)
    runmem5_bytes = end_mem5 - end_mem4

    # Compute deceptive almost-sure winning states of hypergame
    dasw = DASW(hg=hypergame)   #Replace this with the construct function, if needed
    dasw.solve()
    end_time6 = time.time()
    end_mem6 = h.heap().size
    runtime6_ms = round((end_time6 - end_time5) * 1e3, ndigits=4)
    runmem6_bytes = end_mem6 - end_mem5
    
    timeconsuming = [runtime1_ms, runtime2_ms, runtime3_ms, runtime4_ms, runtime5_ms, runtime6_ms]
    spaceconsuming = [runmem1_bytes, runmem2_bytes, runmem3_bytes, runmem4_bytes, runmem5_bytes, runmem6_bytes]
    
    #Return the list of time and space consuming for [BDD, game, igraph, hypergame, SW, DASW]
    return timeconsuming, spaceconsuming
    
if __name__ == '__main__':
    start_time = time.time()
    h = guppy.hpy()
    start_mem = h.heap().size
    # Define BDD
    bdd = BDD()
    end_time1 = time.time()
    end_mem1 = h.heap().size
    runtime1_ms = round((end_time1 - start_time) * 1e3, ndigits=4)
    runmem1_bytes = end_mem1 - start_mem

    # Game object
    game, game_states, game_actions = construct_game(bdd)
    end_time2 = time.time()
    end_mem2 = h.heap().size
    runtime2_ms = round((end_time2 - end_time1) * 1e3, ndigits=4)
    runmem2_bytes = end_mem2 - end_mem1

    # Inference graph object
    igraph, igraph_states, igraph_actions = construct_igraph(bdd, game_actions)
    end_time3 = time.time()
    end_mem3 = h.heap().size
    runtime3_ms = round((end_time3 - end_time2) * 1e3, ndigits=4)
    runmem3_bytes = end_mem3 - end_mem2

    # Product computation
    hypergame = product(bdd, game, igraph)
    end_time4 = time.time()
    end_mem4 = h.heap().size
    runtime4_ms = round((end_time4 - end_time3) * 1e3, ndigits=4)
    runmem4_bytes = end_mem4 - end_mem3

    # Compute sure winning states of hypergame
    sw = SW(hg=hypergame)
    sw.solve()
    end_time5 = time.time()
    end_mem5 = h.heap().size
    runtime5_ms = round((end_time5 - end_time4) * 1e3, ndigits=4)
    runmem5_bytes = end_mem5 - end_mem4

    # Compute deceptive almost-sure winning states of hypergame
    dasw = DASW(hg=hypergame)
    dasw.solve()
    end_time6 = time.time()
    end_mem6 = h.heap().size
    runtime6_ms = round((end_time6 - end_time5) * 1e3, ndigits=4)
    runmem6_bytes = end_mem6 - end_mem5
    
    # Complete notification
    print("ok")
