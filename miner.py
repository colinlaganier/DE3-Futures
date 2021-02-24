
class Miner:

    def __init__(self, proof_of_work):
        self.proof_of_work = proof_of_work
        self.proof_of_work.register(self)

    def mine(self):
        pass

    def notify(self, *args, **kwargs):
        pass