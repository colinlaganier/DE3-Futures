from multiprocessing import Process

from blockchain import BlockChainFactory
from vote import Vote
from legislation import Legislation
from misc import MPQueueWrapper

def main():
    
    # Create Multithreaded Queues for legislation
    voter_queue = MPQueueWrapper()
    legislation_queue = MPQueueWrapper()

    # Initialise the blockchain
    blockchainfactory = BlockChainFactory(voter_queue, legislation_queue)
    blockchain = blockchainfactory.blockchain

    # Update outputs to the correct queues
    Vote.set_output(voter_queue)
    Legislation.set_output(legislation_queue)

    

    for voter in range(10):
        client = AutoClient()


if __name__ == "__main__":
    main()