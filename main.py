from multiprocessing import Process

from blockchain import BlockChainFactory
from miner import Miner
from vote import Vote
from legislation import Legislation
from misc import MPQueueWrapper

def main():
    
    # Create Multithreaded Queues for legislation
    voter_queue = MPQueueWrapper()
    legislation_queue = MPQueueWrapper()

    # Update outputs to the correct queues
    Vote.set_output(voter_queue)
    Legislation.set_output(legislation_queue)

    # Initialise the blockchain
    blockchainfactory = BlockChainFactory(voter_queue, legislation_queue)
    blockchain = blockchainfactory.blockchain
    proof_of_work = blockchainfactory.proof_of_work
    
    mine_q = MPQueueWrapper()
    miner = Miner(proof_of_work)
    mine_process = Process(None, miner, args=(mine_q))    



    for voter in range(10):
        client = AutoClient()


if __name__ == "__main__":
    main()