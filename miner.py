from random import randrange
from misc import TaxToken
from block import UnvalidatedBlock

class Miner:

    def __init__(self, proof_of_work, queue):
        self.queue = queue
        proof_of_work.register_miner(self, queue)
        self.max_nonce = proof_of_work.max_nonce
        self.validation_queue = proof_of_work.validation_queue
        self._tokens = set()

    def __call__(self):
        for nonse in self.mine_continuous():
            self.validation_queue.push((nonse, self))

    def mine_continuous(self):
        while True:
            nonse = self.mine()
            if isinstance(nonse, int):
                yield nonse
            else:
                self.mine(nonse)


    def mine(self, unvalidated_block):
        while True:
            if not self.queue.empty:
                item = self.queue.pop()
                if isinstance(item, TaxToken):
                    self._token.add(TaxToken)
                elif isinstance(item, UnvalidatedBlock):
                    unvalidated_block = item

            nonse = randrange(0, self.max_nonse)
            if int(block.hash, 16) < 2**(256-self.moderator.difficulty):
                return nonse
        
        return None

    def receive_token(self, token):
        self._tokens.add(token)
