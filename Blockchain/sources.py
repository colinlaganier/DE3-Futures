import abc
from legislation import Legislation
from vote import Vote, EncryptedVote

class Source(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def has_block(self):
        pass

    @abc.abstractmethod
    def get_block(self):
        pass


class LegislationSource(Source):
    
    def __init__(self, queue, items_per_block=10):
        self._queue = queue
        self._items_per_block = items_per_block
        self._queue_size = 0

    def has_block(self):
        if self._queue_size >= self._items_per_block:
            return True
        else:
            return False

    def get_block(self):
        if self.has_block():
            data = [self._queue.pop() for _ in range(self._items_per_block)]
            self._queue_size -= self._items_per_block
            return data
        else:
            raise ValueError("No valid block")
    
    def push(self, legislation):
        if issubclass(legislation, Legislation):
            self._queue.push(legislation)
            self._queue_size += 1
        else:
            raise TypeError("Invalid legislation object")

class VoteSource(Source):

    def __init__(self, queue, items_per_block=50):
        self._queue = queue
        self._items_per_block = items_per_block
        self._queue_size = 0

    def has_block(self):
        if self._queue_size >= self._items_per_block:
            return True
        else:
            return False

    def get_block(self):
        if self.has_block():
            data = [self._queue.pop() for _ in range(self._items_per_block)]
            self._queue_size -= self._items_per_block
            return data
        else:
            raise ValueError("No valid block")
    
    def push(self, legislation):
        if issubclass(legislation, Vote):
            self._queue.push(legislation)
            self._queue_size += 1
        else:
            raise TypeError("Invalid legislation object")