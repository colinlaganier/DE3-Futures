import hashlib
from datetime import datetime
from functools import reduce

from vote import Vote
from legislation import Legislation

# TOD
#   > Automatically updating hash on change
#   > What to do if there are insufficient votes for a whole block?
#   > 
#

class Moderator:

    def __init__(self, time_per_block=300, blocks_per_chunk=1024):
        self.block_store = []
        self._difficulty
        self._time_per_block = time_per_block
        self._blocks_per_chunk = blocks_per_chunk

    @property
    def difficulty(self):
        return self.difficulty

    @difficulty.setter
    def difficulty(self, value):
        raise PermissionError()
    
    def notify(self, block):
        self.block_store.append(block)

        if len(self.block_store) == self._blocks_per_chunk:
            self._difficulty = self.recalculate_difficulty()
            self.block_store = []

    def recalculate_difficulty(self):
        time_taken = (self.block_store[-1] - self.block_store[0]).total_seconds()
        if time_taken > self._time_per_block:
            self._difficulty += 1
        else:
            self._difficulty -= 1

class ProofOfWork:

    def __init__(self, blockchain):
        self._blockfactory = BlockFactory()
        self.blockchain = blockchain
        self._next_block = blockfactory.next_block()

        self.moderator = blockchain.moderator
        self.register(self.moderator)

        self.subscribed = []

    @property
    def unvalidated_block(self):
        return self._next_block
    
    @unvalidated_block.setter
    def unvalidated_block(self, value):
        raise PermissionError("Can't change value of next block")

    def validate_block(self, nonse):
        block = self.get_block_from_unvalidated_block(nonse)

        if int(block.hash, 16) < 2**(256-self.moderator.difficulty):
            self.blockchain.add(block)
            self.notify_all(block)
            return True

        return False

    def get_block_from_unvalidated_block(self, nonse):
        data = self._next_block.data
        prev_hash = self._next_block.previous_hash
        block_id = self._next_block.block_id
        time_stamp = self._next_block.time_stamp

        return Block(data, prev_hash, nonse, block_id, time_stamp)

    def notify_all(self, block):
        for sub in self.subscribed:
            sub.notify(block)
        
    def register(self, object):
        has_notify = getattr(object, "notify", None)
        if callable(has_notify):
            self.subscribed.append(object)
        else:
            raise ValueError("Invalid object")
    
    def deregister(self, object):
        if object in self.subscribed:
            self.subscribed.remove(object)

    def register_source(self, source, priority):
        self._blockfactory.register_source(source, priority)



class BlockFactory:

    def __init__(self, proof_of_work, votes_per_block):
        self.votes_per_block = votes_per_block
        proof_of_work.register(self)
        self._unvalidated_block = self._get_next_block()
        self._sources = []

    @property
    def next_block(self):
        return self._unvalidated_block

    @next_block.setter
    def next_block(self, value):
        raise PermissionError("Can't change this")

    def _get_next_block(self, previous_hash):

        for priority, source in sorted(self._sources, key=lambda x: x[0]):
            while True:
                if source.has_block():
                    data = source.get_block()
                    return UnvalidatedBlock(data, previous_hash)

    def register_source(self, source, priority=100):
        if issubclass(source, Source):
            self._sources.append(priority, source)

    def notify(self, block):
        if block.is_valid():
            self._unvalidated_block = self._get_next_block(block.hash)


class UnvalidatedBlock:

    count = 0

    def __init__(self, data, previous_hash):
        self.previous_hash = previous_hash
        self.data = data
        self.block_id = UnvalidatedBlock.count + 1
        self.time_stamp = datetime.now()

    def make_block(self, nonce):
        block = Block(self.data, self.previous_hash, nonce, self.block_id, self.time_stamp)
        if block.is_valid():
            return block
        else:
            raise ValueError("Block created was invalid")

class Block:

    def __init__(self, data, previous_hash, nonce, block_id, time_stamp=None, hash_function=hashlib.sha256):
        self.previous_hash = previous_hash
        self.data = data
        self.block_id = block_id

        self.nonce = nonce

        if time_stamp is None:
            self.time_stamp = datetime.now()
        else:
            self.time_stamp = time_stamp
        self._next = None

        self.hash_function = hash_function
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        hasher = self.hash_function()
        prepped_data = self.prep_hash()
        hasher.update(prepped_data)

        return hasher.hexdigest()

    def update_hash(self):
        self.hash = self.calculate_hash()

    def prep_hash(self):
        hashing_data = [self.data, self.block_id, self.nonce, self.previous_hash, self.time_stamp]
        hashing_data = [str(x).encode("utf-8") for x in hashing_data]
        return reduce(lambda x, y: x+y, hashing_data)

    @property
    def next_block(self):
        return self._next
    
    @next_block.setter
    def next_block(self, new_block):
        if isinstance(new_block, Block):
            self._next = new_block
        else:
            raise ValueError("Invalid Block")
    
    def __str__(self):
        return f"Block {self.block_id} spawned from {self.previous_hash}"