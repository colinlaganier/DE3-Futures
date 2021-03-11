import hashlib
from datetime import datetime
from functools import reduce

from vote import Vote
from legislation import Legislation
from misc import TaxToken, MPQueueWrapper
from sources import Source

# TOD
#   > Automatically updating hash on change
#   > What to do if there are insufficient votes for a whole block?
#   > 
#

class Moderator:

    def __init__(self, time_per_block=1.21E6, blocks_per_chunk=1024):
        """ Creates moderator object that adjusts the difficulty of 
            the proof of work algorithm """
        self.block_store = []
        self._difficulty = 1
        self._time_per_block = time_per_block
        self._blocks_per_chunk = blocks_per_chunk

    # Ensures difficulty property cant be changed by outside users
    @property
    def difficulty(self):
        return self._difficulty

    @difficulty.setter
    def difficulty(self, value):
        raise PermissionError()
    
    def notify(self, block):
        """ Receives notification when new block and adjusts difficulty """
        self.block_store.append(block)

        if len(self.block_store) == self._blocks_per_chunk:
            self._difficulty = self.recalculate_difficulty()
            self.block_store = []

    def recalculate_difficulty(self):
        """ Simple adjustment of difficulty based on the time taken to mine a number of blocks"""
        time_taken = (self.block_store[-1].time_stamp - self.block_store[0].time_stamp).total_seconds()
        if time_taken > self._time_per_block:
            self._difficulty += 1
        else:
            self._difficulty -= 1


class ProofOfWork:

    def __init__(self, blockchain):
        """ Manages the addition of blocks to the blockchain using a SHA256 
            proof of work scheme """
        self.blockchain = blockchain

        self.moderator = blockchain.moderator

        self.validation_queue = MPQueueWrapper()
        self.miner_manager_sq = None
        self.miner_manager_rq = None

        self.subscribed = []

        self.register(self.moderator)

        self._blockfactory = BlockFactory(self, 50)
        self._next_block = self._blockfactory.next_block

    # Ensures the current unvalidated block is immutable
    @property
    def unvalidated_block(self):
        return self._next_block
    
    @unvalidated_block.setter
    def unvalidated_block(self, value):
        raise PermissionError("Can't change value of next block")

    # Ensures the maximum nonce value is immutable
    @property
    def max_nonce(self):
        return self.blockchain.max_nonce

   
    def add_miner_manager(self, send_q, receive_q):
        """ Conned the proof of work to the miner manager through threadsafe queues """
        self.miner_manager_sq = send_q
        self.miner_manager_rq = receive_q

    def listen(self):
        """ Listens for data requests and nonse validation requests before processing them """
        mm_message_functions = {"GETCOND":self.send_conditions,
                                "GETBLOCK":self.send_block}

        while True:
            if not self.validation_queue.empty():
                item = self.validation_queue.pop()
                if item is None:
                    break
                else:
                    self.validate_block(**item)
            if not self.miner_manager_rq.empty():
                item = self.miner_manager_rq.pop()
                mm_message_functions[item[0]](**item[1])

    def send_conditions(self, id=None):
        """ Sends requested conditions to miner manager """
        difficulty = self.moderator.difficulty
        self.miner_manager_sq.push(("UPDATE",{"id":id,
                                              "difficulty":difficulty,
                                              "max_nonse":self.blockchain.max_nonce}))

    def send_block(self, id=None):
        """ Sends requested block to miner manager """
        self.miner_manager_sq.push("NEW", {"block":self._next_block,
                                           "id":id})
    
    def validate_block(self, nonse, miner):
        """ Checks to see if the block is valid based on the given nonce """
        block = self.get_block_from_unvalidated_block(nonse)

        if int(block.hash, 16) < 2**(256-self.moderator.difficulty):
            self.blockchain.add(block)
            self.notify_all(block)
            self._next_block = self._blockfactory.next_block()
            self.miner_manager_sq.push(("TOKEN", {"token":TaxToken(), 
                                                  "id":miner}))
            self.send_new_block()
            return True

        return False

    def get_block_from_unvalidated_block(self, nonse):
        """ Utility function to make a Block from an UnvalidatedBlock """
        data = self._next_block.data
        prev_hash = self._next_block.previous_hash
        block_id = self._next_block.block_id
        time_stamp = self._next_block.time_stamp

        return Block(data, prev_hash, nonse, block_id, time_stamp)

    def notify_all(self, block):
        """ Notifies all subscribers that a block has been mined """
        for sub in self.subscribed:
            sub.notify(block)

    def send_new_block(self): 
        """ Pushes a new block to the miner manager """
        self.miner_manager_sq.push(("NEW",{"block":self._next_block}))
        
    def register(self, object):
        """ Registers objects for notifications """
        has_notify = getattr(object, "notify", None)
        if callable(has_notify):
            self.subscribed.append(object)
        else:
            raise ValueError("Invalid object")
    
    def deregister(self, object):
        """ Deregisters for notifications """
        if object in self.subscribed:
            self.subscribed.remove(object)

    def register_source(self, source, **kwargs):
        """ Adds a source for the blockfactory """
        self._blockfactory.register_source(source, **kwargs)


class BlockFactory:

    def __init__(self, proof_of_work, votes_per_block):
        self.votes_per_block = votes_per_block
        proof_of_work.register(self)
        self._sources = []

        self._unvalidated_block = self._get_next_block(proof_of_work.blockchain.tail.hash)

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
        if isinstance(source, Source):
            self._sources.append((priority, source))

    def notify(self, block):
        self._unvalidated_block = self._get_next_block(block.hash)


class UnvalidatedBlock:

    count = 0

    def __init__(self, data, previous_hash):
        """ Data structure that contains information for a Block without a nonce """
        self.previous_hash = previous_hash
        self.data = data
        self.block_id = UnvalidatedBlock.count + 1
        self.time_stamp = datetime.now()

    def make_block(self, nonce):
        """ Creates a Block from the contained data """
        block = Block(self.data, self.previous_hash, nonce, self.block_id, self.time_stamp)
        if block.is_valid():
            return block
        else:
            raise ValueError("Block created was invalid")


class Block:

    def __init__(self, data, previous_hash, nonce, block_id, time_stamp=None, hash_function=hashlib.sha256):
        """ Container that is used with the blockchain. Defaults to using a sha256 hash function - this should be 
            the same as used by the proof of work validation function """
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
        """ Calculates the block hash based on the concatenation of contained data """
        hasher = self.hash_function()
        prepped_data = self.prep_hash()
        hasher.update(prepped_data)

        return hasher.hexdigest()

    def update_hash(self):
        """ Recalculates the hash, used for readability """
        self.hash = self.calculate_hash()

    def prep_hash(self):
        """ Utility function for concatenating data """
        hashing_data = [self.data, self.block_id, self.nonce, self.previous_hash, self.time_stamp]
        hashing_data = [str(x).encode("utf-8") for x in hashing_data]
        return reduce(lambda x, y: x+y, hashing_data)

    # Ensures blocks are set correctly
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