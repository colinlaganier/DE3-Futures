from random import randrange
from misc import TaxToken
from block import UnvalidatedBlock
from misc import MPQueueWrapper

class Miner:

    def __init__(self, manager):
        self.queue = queue
        
        self.id, self.validation_queue = manager.register_miner(self.queue)
        manager.get_conditions(self.id)

        self.message_functions = {"NEW":self.new_block,
                                  "UPDATE":self.update_conditions,
                                  "TOKEN":self.receive_token}
        self._stop = 0

        self.unvalidated_block = None
        self.difficulty = 200
        self.max_nonse = 2**32

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
        # TODO MAKE THIS WORK
        while True:
            if not self.queue.empty:
                item = self.queue.pop()
                self.message_functions[item[0]](**item[1])

            if self._stop:
                self._stop = 0
                return None

            elif unvalidated_block is None:
                continue

            nonse = randrange(0, self.max_nonse)
            if int(block.hash, 16) < 2**(256-self.difficulty):
                return nonse
        
        return None

    def new_block(self, block):
        self._stop = 1
        self.unvalidated_block = block

    def update_conditions(self, difficulty=None, max_nonse=None):
        if difficulty is not None:
            self.difficulty = difficulty
        if max_nonse is not None:
            self.max_nonse = max_nonse

    def receive_token(self, token):
        self._tokens.add(token)


class MinerManager:

    def __init__(self, validation_queue):
        self.validation_queue = validation_queue

        self.proof_of_work_send = MPQueueWrapper()
        self.proof_of_work_receive = MPQueueWrapper()

        self.miner_qs = {}
        self.miner_count = 0
    
    def connect(self, proof_of_work):
        proof_of_work.add_miner_manager(self.proof_of_work_receive, 
                                        self.proof_of_work_send)

    def __call__(self):
        message_received_functions = {"NEW":self.new_block,
                                      "UPDATE": self.update_conditions,
                                      "TOKEN": self.send_reward}
        while True:
            if not self.proof_of_work_receive.empty():
                item = self.input_q.pop()
                message_received_functions[item[0]](**item[1])

    def new_block(self, block=None, id=None):
        if id is None:
            for queue in self.miner_qs.values():
                queue.push(("NEW", {"block":block}))
        else:
            self.miner_qs[id].push(("NEW", {"block":block}))

    def update_conditions(self, id=None, conditions=None):
        if id is None:
            for queue in self.miner_qs.values():
                queue.push({"UPDATE":conditions})
        else:
            self.miner_qs[id].push(("UPDATE", {"conditions":conditions}))

    def send_reward(self, id=None, token=None):
        self.miner_qs[id].push(("TOKEN", {"token":token}))

    def get_conditions(self, id):
        self.proof_of_work_q.push(("GETCOND", {"id":id}))
    
    def get_block(self, id):
        self.proof_of_work_send(("GETBLOCK", {"id":id}))

    def register_miner(self, queue):
        self.miner_count += 1
        self.miner_qs[self.miner_count] = queue
        return (self.miner_count, validation_queue)

    def deregister_miner(self, queue):
        self.miner_qs.remove(queue)