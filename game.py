import dd


class Game:
    def __init__(self, num_states, num_actions):
        self.bdd_states = dd.BDD()
        self.bdd_actions = dd.BDD()
        self.bdd_trans = dd.BDD()

    def add_state(self, uid):
        pass

    def add_trans(self, uid, aid, vid):
        pass

    def is_valid_state(self, uid):
        pass

    def is_valid_action(self, aid):
        pass

    def is_valid_trans(self, uid, aid, vid):
        pass

    def succ(self, uid, aid=None):
        pass

    def pred(self, vid, aid=None):
        pass

    