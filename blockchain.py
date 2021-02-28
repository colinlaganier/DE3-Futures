from block import Block, Moderator, ProofOfWork
from time import perf_counter
from sources import LegislationSource, VoteSource

class BlockChain:

    def __init__(self, max_nonce=2**32):
        self.genesis = Block("", "Genesis", 0, 0)
        self.tail = self.genesis
        self.moderator = Moderator(time_per_block=300)

        self.max_nonce = max_nonce

    def add(self, data, nonce, time_stamp):
        block = Block(data, self.tail.hash, nonce, self.tail.block_id+1, time_stamp=time_stamp)
    
        if int(block.hash, 16) < 2**(256-self.moderator.difficulty):
            self.tail.next_block = block
            self.tail = block
        else:
            raise ValueError("INVALID BLOCK")
    

class BlockChainFactory:

    def __init__(self, voter_input, legislation_input):
        self.blockchain = BlockChain()
        self.proof_of_work = ProofOfWork(self.blockchain)

        commons_source = LegislationSource(legislation_input)
        vote_source = VoteSource(voter_input)

        self.proof_of_work.register_source(commons_source)
        self.proof_of_work.register_source(vote_source)

        self.proof_of_work.notify_all(None)

if __name__ == "__main__":

    pass
