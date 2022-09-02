import multiprocessing
import multiprocessing.queues as queues

class SimpleQueue:

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
 
class MPQueueWrapper(queues.Queue):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ctx=multiprocessing.get_context())

    def push(self, data, *args, **kwargs):
        self.put(data, *args, **kwargs)

    def pop(self, *args, **kwargs):
        return self.get(*args, **kwargs)


class MLPipelineFake:

    def __init__(self):
        pass

    def encode(self, comments):
        return " ".join(["ENCODED COMMENTS", comments, "ENCODED COMMENTS"])


class TaxToken:

    count = 0

    def __init__(self):
        self.id = TaxToken.count
        TaxToken.count += 1

    def __hash__(self):
        return hash(str(self.id))
