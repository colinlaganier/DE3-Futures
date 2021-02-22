class Queue:

    def __init__(self):
        self._queue = []
        self.size = 0
    
    def push(self, data):
        self._queue.append(data)
        self.size += 1

    def pop(self):
        if self._queue:
            rtn = self._queue[0]
            self._queue = self._queue[1:]
            self.size -= 1
            return rtn
        else:
            return None
    
    def is_empty(self):
        return bool(self._queue)
 