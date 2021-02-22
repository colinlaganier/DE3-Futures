from misc import Queue

class Legislation:

    queue = Queue()
    count = 0

    def __init__(self, name, authors, contents):
        self.id = Legislation.count
        Legislation.count += 1

        self.name = name
        self.authors = authors
        self.contents = contents

        self.registered_voters = []

    def register(voter):
        self.registered_voters.append(voter)

    def notify(self):
        """ Notify voters to submit their vote """
        for voter in self.registered_voters:
            #def notify(self, legislation, for_legislation, comments, level, final_call=0):
            voter.notify(self, None, None, None, final_call=1)
    
    def put_to_public(self):
        Legislation.queue.push(self)


class HealthLegislation(Legislation):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EpidemicLegislation(HealthLegislation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)