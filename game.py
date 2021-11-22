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
        self.bdd_actions.declare(*[f"a{i}" for i in range(self.bits_actions)])
        self.bdd_trans.declare(*[f"u{i}" for i in range(self.bits_states)],
                               *[f"a{i}" for i in range(self.bits_actions)],
                               *[f"v{i}" for i in range(self.bits_states)]
                               )

        self.state_validity_formula = None
        self.action_validity_formula = None
        self.trans_validity_formula = None

        logger.info(f"bdd_states.vars={self.bdd_states.vars}.")
        logger.info(f"bdd_actions={self.bdd_actions.vars}.")
        logger.info(f"bdd_trans={self.bdd_trans.vars}.")

    def add_state(self, uid):
        assert uid < 2**self.bits_states

        # Represent uid as expression using bit variables
        uid_expr = self._uid2expr(uid)
        logger.debug(f"uid: {uid}, expr: {uid_expr}")

        # Update state validity expression
        f = self.bdd_states.add_expr(uid_expr)
        if self.state_validity_formula is None:
            self.state_validity_formula = f
        else:
            self.state_validity_formula |= f

    def add_action(self, aid):
        assert aid < 2 ** self.bits_states

        # Represent uid as expression using bit variables
        aid_expr = self._aid2expr(aid)
        logger.info(f"aid: {aid}, expr: {aid_expr}")

        # Update action validity expression
        f = self.bdd_actions.add_expr(aid_expr)
        if self.action_validity_formula is None:
            self.action_validity_formula = f
        else:
            self.action_validity_formula |= f

    def add_trans(self, uid, aid, vid):
        assert self.is_valid_state(uid) and self.is_valid_state(vid) and self.is_valid_action(aid)

        # Convert input uid to binary format
        uid_expr = self._uid2expr(uid, varname="u")
        vid_expr = self._uid2expr(vid, varname="v")
        aid_expr = self._aid2expr(aid)
        trans_expr = " & ".join([uid_expr, aid_expr, vid_expr])

        logger.info(f"uid: {uid}, uid_expr: {uid_expr}")
        logger.info(f"vid: {vid}, vid_expr: {vid_expr}")
        logger.info(f"aid: {aid}, aid_expr: {aid_expr}")
        logger.info(f"trans_expr: {trans_expr}")

        # Update state validity expression
        f = self.bdd_trans.add_expr(trans_expr)
        if self.trans_validity_formula is None:
            self.trans_validity_formula = f
        else:
            self.trans_validity_formula |= f

    def is_valid_state(self, uid):
        # Construct map of variables to valuation
        uid_dict = self._uid2dict(uid)

        # Evaluate state validity expression
        val = self.bdd_states.let(uid_dict, self.state_validity_formula)
        if val.to_expr() == 'FALSE':
            return False
        elif val.to_expr() == 'TRUE':
            return True
        else:
            raise ValueError("Unknown value")

    def is_valid_action(self, aid):
        # Construct map of variables to valuation
        aid_dict = self._aid2dict(aid)

        # Evaluate state validity expression
        val = self.bdd_actions.let(aid_dict, self.action_validity_formula)
        if val.to_expr() == 'FALSE':
            return False
        elif val.to_expr() == 'TRUE':
            return True
        else:
            raise ValueError("Unknown value")

    def is_valid_trans(self, uid, aid, vid):
        # Construct map of variables to valuation
        uid_dict = self._uid2dict(uid, varname="u")
        vid_dict = self._uid2dict(vid, varname="v")
        aid_dict = self._aid2dict(aid, varname="a")

        trans_dict = dict()
        trans_dict.update(uid_dict)
        trans_dict.update(aid_dict)
        trans_dict.update(vid_dict)

        # Evaluate state validity expression
        val = self.bdd_trans.let(trans_dict, self.trans_validity_formula)
        if val.to_expr() == 'FALSE':
            return False
        elif val.to_expr() == 'TRUE':
            return True
        else:
            raise ValueError("Unknown value")

    def succ(self, uid, aid=None):
        pass

    def pred(self, vid, aid=None):
        pass

    def _uid2binary(self, uid):
        return format(uid, f"0{self.bits_states}b")

    def _uid2expr(self, uid, varname="u"):
        uid_binary = self._uid2binary(uid)
        expr = [f"{varname}{i}" if uid_binary[-1 - i] == "1" else f"~{varname}{i}" for i in range(0, self.bits_states)]
        expr.reverse()
        expr = " & ".join(expr)
        return expr

    def _uid2dict(self, uid, varname="u"):
        uid_binary = self._uid2binary(uid)
        dict_ = dict()
        for i in range(0, self.bits_states):
            if uid_binary[-1 - i] == "1":
                dict_[f"{varname}{i}"] = True
            else:
                dict_[f"{varname}{i}"] = False
        return dict_

    def _aid2binary(self, aid):
        return format(aid, f"0{self.bits_actions}b")

    def _aid2expr(self, aid, varname="a"):
        aid_binary = self._aid2binary(aid)
        expr = [f"{varname}{i}" if aid_binary[-1 - i] == "1" else f"~{varname}{i}" for i in range(0, self.bits_actions)]
        expr.reverse()
        expr = " & ".join(expr)
        return expr

    def _aid2dict(self, aid, varname="a"):
        aid_binary = self._aid2binary(aid)
        dict_ = dict()
        for i in range(0, self.bits_actions):
            if aid_binary[-1 - i] == "1":
                dict_[f"{varname}{i}"] = True
            else:
                dict_[f"{varname}{i}"] = False
        return dict_


if __name__ == '__main__':
    game = Game(3, 2)
    game.add_state(0)
    game.add_state(1)
    print(f"game.is_valid_state(0)={game.is_valid_state(0)}")
    print(f"game.is_valid_state(2)={game.is_valid_state(2)}")
    game.add_state(2)
    print("Added state: 2")
    print(f"game.is_valid_state(2)={game.is_valid_state(2)}")

    game.add_action(0)
    game.add_action(1)

    game.add_trans(0, 0, 1)
    print(f"game.is_valid_trans(0, 0, 1)={game.is_valid_trans(0, 0, 1)}")
    print(f"game.is_valid_trans(0, 0, 2)={game.is_valid_trans(0, 0, 2)}")


