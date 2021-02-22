import abc
from random import randint

from misc import Queue

from legislation import Legislation

# TODO:
#   > Add deregistration of voters   

class Encrypter(abc.ABC):

    @abc.abstractmethod
    def __init__(self, salt):
        self.salt = salt
    

    @abc.abstractmethod
    def decrypt(self, key):
        raise NotImplementedError()

    def encrypt(self, vote, key):
        if self.validate(vote, key):
            data = self._get_cipher_text(vote, key)
            return EncryptedVote(data)
        raise ValueError()

    @abc.abstractmethod
    def _get_cipher_text(self, vote, key):
        raise NotImplementedError()


class FakeEncrypter(Encrypter):

    def decrypt(self, vote, key):
        return "CANT DO THIS RIGHT NOW"

    def _get_cipher_text(self, vote, key):
        return "ENCRYPTED VOTE: " + str(vote)


class Vote:

    vote_queue = Queue()

    def __init__(self, legislation, vote_for, voter, comments):
        if isinstance(legislation, Legislation):
            self.legislation = legislation
        else:
            raise TypeError("Invalid Legislation Object")

        if isinstance(vote_for, bool):
            self.vote_for = vote_for
        else:
            raise TypeError("Invalid Vote")

        if isinstance(voter, Voter):
            self.voter = voter
        else:
            raise TypeError("Invalid Voter")
        self.comments = comments

    def cast(self, key, encrypter=FakeEncrypter):
        # ------------------------------------------------------------------
        #   Needs some work doing. Pipe comments to ML system
        # ------------------------------------------------------------------ #
        encrypter = Encrypter(randint(0, 2**32))
        encrytped_vote = encrypter.encrypt(self, key)
        Vote.vote_queue


    
    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return " ".join([str(x) for x in (self.legislation, self.vote_for, self.voter)])


class EncryptedVote:

    def __init__(self, data):
        self._data

    def __str__(self):
        return str(data)
    
    def __hash__(self):
        return hash(self.data)


class Voter:

    def __init__(self, id, district, public_key):
        self.id = id
        self.district = district
        self.public_key = public_key

        self.subscribed_to_me = dict()
        self.notifications = dict()
        self.proxy = VoterProxy(self)

    def vote(self, legislation, for_legislation, comments):
        self.notify_all_subs(legislation, for_legislation, comments)
        vote = Vote(legislation, for_legislation, self, comments)
        vote.cast(self.public_key)

    def notify_all_subs(self, legislation, for_legislation, comments, level=0):
        legislative_list = [type(legislation)]
        while legislative_list:
            legislative_type = legislative_list.pop()
            if legislative_type.__name__ in self.subscribed_to_me:
                for sub in self.subscribed_to_me[legislative_type.__name__]:
                    sub.notify(legislation, for_legislation, comments, level)
                
            parent_type = legislative_type.__bases__
            if issubclass(parent_type, Legislation):
                legislative_list.append(parent_type)
            level += 1
    
    def subscribe(self, voter, legislation_type):
        if not issubclass(legislation_type, Legislation):
            raise ValueError
        try:
            voter.register(self.proxy, legislation_type)
        except KeyError:
            voter.register(self.proxy, legislation_type)

    def register(voter, legislation_type):
        try:
            self.subscribed_to_me[legislation_type.__name__].append(voter)
        except KeyError:
            self.subscribed_by_me[legislation_type.__name__] = [voter]


    def notify(self, legislation, for_legislation, comments, level, final_call=0):
        if final_call:
            self.vote(*self.notifications[legislation.id][1])
            del self.notifications[legislation.id]
            return
        self.notify_all_subs(legislation, for_legislation, comments, level)

        try:
            if self.notifications[legislation.id][0] > level:
                self.notifications[legislation.id] = (level, Vote(legislation, for_legislation, self, comments))
        except KeyError:
            self.notifications[legislation.id] = (level, Vote(legislation, for_legislation, self, comments))
            legislation.register(self.proxy)


class VoterProxy:

    def __init__(self, voter):
        self._voter = voter

    def notify(self, *args, **kwargs):
        self._voter.notify(*args, **kwargs)

    def register(voter, legislation_type):
        self._voter.register(voter, legislation_type)


