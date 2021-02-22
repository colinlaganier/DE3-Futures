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
    
    def __init__(self):
        pass


class VoteSource(Source):

    def __init__(self):
