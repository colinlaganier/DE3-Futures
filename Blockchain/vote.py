import abc
from random import randint

from misc import SimpleQueue, MLPipelineFake

from legislation import Legislation

# TODO:
#   > Add deregistration of voters   

class Encrypter(abc.ABC):

    @abc.abstractmethod
    def __init__(self, salt):
        """ Abstract interface required by encrypter objects in the system """
        self.salt = salt
    
    # Defines required methods that Encypters need to implement
    @abc.abstractmethod
    def decrypt(self, key):
        raise NotImplementedError()

    def encrypt(self, data, key):
        cipher_text = self._get_cipher_text(data, key)
        return EncryptedVote(cipher_text)

    @abc.abstractmethod
    def _get_cipher_text(self, data, key):
        raise NotImplementedError()


class FakeEncrypter(Encrypter):
    """ A fake Encrypter that acts as an encrypter object without 
        implementing the encryption algorithm """
    def decrypt(self, data, key):
        return str(data)

    def _get_cipher_text(self, data, key):
        return "<ENCRYPTED VOTE>" + str(data) + "<ENCRYPTED VOTE>"


class Vote:

    # Output stream for vote casting
    vote_queue = SimpleQueue()
    # Machine learning pipeline that encodes comments
    ml_pipeline = MLPipelineFake()

    def __init__(self, legislation, vote_for, voter, comments):
        """ Vote object that can be cast onto the blockchain. 
            Votes are first converted to EncryptedVotes prior too being commited """
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

    # Sets vote casting stream
    @classmethod
    def set_output(cls, output):
        cls.queue = output

    def cast(self, key, encrypter=FakeEncrypter):
        """ Casts vote to the blockchain after processing comments and encrypting information. """

        encoded_comments = Vote.ml_pipeline.encode(self.comments)
        encrypter = Encrypter(randint(0, 2**32))

        plain_text = f"{Legislation.id}{self.vote_for}{encoded_comments}"
        cipher_text = f"{str(self.voter)}{self.comments}"
        cipher_text = encrypter.encrypt(cipher_text, key)
        encrypted_vote = EncryptedVote(cipher_text, plain_text)
        Vote.vote_queue.push(encrypted_vote)


    
    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return " ".join([str(x) for x in (self.legislation, self.vote_for, self.voter, self.comments)])


class EncryptedVote:

    def __init__(self, cipher_text, plain_text):
        """ Simple container of vote information """
        self.cipher_text = cipher_text
        self.plain_text = plain_text

    def __str__(self):
        return f"CIPHER TEXT: {self.cipher_text}\nPLAIN TEXT: {self.plain_text}"
    
    def __hash__(self):
        return hash(str(self))


class Voter:

    def __init__(self, id, district, public_key):
        """ Container for each individual voter in the system, allowing them to vote, 
            delegate and receive votes. """
        self.id = id
        self.district = district
        self.public_key = public_key

        self.subscribed_to_me = dict()
        self.notifications = dict()
        self.proxy = VoterProxy(self)

    def vote(self, legislation, for_legislation, comments):
        """ Cast vote on a piece of legislation with comments """
        self.notify_all_subs(legislation, for_legislation, comments)
        vote = Vote(legislation, for_legislation, self, comments)
        vote.cast(self.public_key)
        legislation.deregister(self)

    def notify_all_subs(self, legislation, for_legislation, comments, level=0):
        """ Notify all subscribers that you are voting for a particular piece of legislation.
            Passes all required voting information """
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
        """ Core aspect of liquid voting system. 
            Subscribe to a delegate on a particular legislation type """
        if not issubclass(legislation_type, Legislation):
            raise ValueError
        try:
            voter.register(self.proxy, legislation_type)
        except KeyError:
            voter.register(self.proxy, legislation_type)

    def register(voter, legislation_type):
        """ Registers subscribers based on what they subscribe to you for """
        try:
            self.subscribed_to_me[legislation_type.__name__].append(voter)
        except KeyError:
            self.subscribed_by_me[legislation_type.__name__] = [voter]

    def deregister(voter, legislation_type):
        """ Removes subscribers """
        try:
            self.subscribed_to_me[legislation_type.__name__].remove(voter)
        except KeyError:
            return

    def notify(self, legislation, for_legislation, comments, level, final_call=0):
        """ Receive notification about voting information. This could be from delegates or a final voting call """
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

    def __str__(self):
        return self.id


class VoterProxy:

    def __init__(self, voter):
        """ Proxy object that limits calls to a Voter object """
        self._voter = voter

    def notify(self, *args, **kwargs):
        self._voter.notify(*args, **kwargs)

    def register(voter, legislation_type):
        self._voter.register(voter, legislation_type)


