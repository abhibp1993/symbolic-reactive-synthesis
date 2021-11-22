import dd
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Game:
    def __init__(self, num_states, num_actions):
        # Determine size of bit-vectors
        self.bits_states = math.ceil(math.log2(num_states))
        self.bits_actions = math.ceil(math.log2(num_actions))
        self.bits_trans = math.ceil(math.log2(num_states * num_actions * num_states))

        logger.info(f"num_states={num_states} using {self.bits_states} bits.")
        logger.info(f"num_actions={num_actions} using {self.bits_actions} bits.")
        logger.info(f"num_trans={num_states**2 * num_actions} using {self.bits_trans} bits.")

        # Initialize BDDss
        self.bdd_states = None
        self.bdd_actions = None
        self.bdd_trans = None
        self.state_validity_formula = None
        self.action_validity_formula = None
        self.trans_validity_formula = None
        self.reset_bdds()

    def reset_bdds(self):
        self.bdd_states = dd.BDD()
        self.bdd_actions = dd.BDD()
        self.bdd_trans = dd.BDD()

        self.bdd_states.declare(*[f"u{i}" for i in range(self.bits_states)])
        self.bdd_actions.declare(*[f"u{i}" for i in range(self.bits_actions)])
        self.bdd_trans.declare(*[f"u{i}" for i in range(self.bits_trans)])

        self.state_validity_formula = None
        self.action_validity_formula = None
        self.trans_validity_formula = None

        logger.info(f"bdd_states.vars={self.bdd_states.vars}.")
        logger.info(f"bdd_actions={self.bdd_actions.vars}.")
        logger.info(f"bdd_trans={self.bdd_trans.vars}.")

    def add_state(self, uid):
        assert uid < 2**self.bits_states

        # Convert input uid to binary format
        uid_binary = format(uid, f"0{self.bits_states}b")

        # Construct expression in terms of bits
        expr = [f"u{i}" if uid_binary[-1 - i] == "1" else f"~u{i}" for i in range(0, self.bits_states)]
        expr.reverse()
        expr = " & ".join(expr)
        logger.info(f"uid: {uid}, uid_binary: {uid_binary} expr: {expr}")

        # Update state validity expression
        f = self.bdd_states.add_expr(expr)
        if self.state_validity_formula is None:
            self.state_validity_formula = f
        else:
            self.state_validity_formula |= f

    def add_trans(self, uid, aid, vid):
        pass

    def is_valid_state(self, uid):
        # Convert input uid to binary format
        uid_binary = format(uid, f"0{self.bits_states}b")

        # Construct map of variables to valuation
        expr = dict()
        for i in range(0, self.bits_states):
            if uid_binary[-1 - i] == "1":
                expr[i] = True
            else:
                expr[i] = False

        # Evaluate state validity expression
        val = self.bdd_states.let(expr, self.state_validity_formula)
        if val.to_expr() == 'FALSE':
            return False
        elif val.to_expr() == 'TRUE':
            return True
        else:
            raise ValueError("Unknown value")

    def is_valid_action(self, aid):
        pass

    def is_valid_trans(self, uid, aid, vid):
        pass

    def succ(self, uid, aid=None):
        pass

    def pred(self, vid, aid=None):
        pass


if __name__ == '__main__':
    game = Game(3, 2)
    game.add_state(0)
    game.add_state(1)
