from misc import SimpleQueue

class Legislation:

    queue = SimpleQueue()
    count = 0
    open_legislation = set()

    def __init__(self, name, authors, contents):
        self.id = Legislation.count
        Legislation.count += 1

        self.name = name
        self.authors = authors
        self.contents = contents

        self.registered_voters = []
        
        Legislation.open_legislation.add(self)

    @classmethod
    def set_output(cls, output):
        cls.queue = output

    def register(voter):
        self.registered_voters.append(voter)

    def deregister(voter):
        self.registered_voters.remove(voter)

    def notify_all(self):
        """ Notify voters to submit their vote """
        for voter in self.registered_voters:
            voter.notify(self, None, None, None, final_call=1)
        Legislation.open_legislation.remove(self)
    
    def put_to_public(self):
        Legislation.queue.push(self)

    def __str__(self):
        return "\n".join([f"{self.id}: {self.name}", self.authors, self.contents])
    
    def __hash__(self):
        return hash(str(self))


class HealthLegislation(Legislation):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EpidemicLegislation(HealthLegislation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
