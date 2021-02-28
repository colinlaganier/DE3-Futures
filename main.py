from multiprocessing import Process

from blockchain import BlockChainFactory
from miner import MinerManager, Miner
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

    miner_manager = MinerManager(proof_of_work.validation_queue)
    miner_manager.connect(proof_of_work)

    miner_processes = []
    for _ in range(3):
        miner = Miner(miner_manager)
        miner_process = Process(None, miner)
        miner_process.start()
        miner_processes.append(miner_process)


    """ client_processes = []
    for voter in range(10):
        client = AutoClient()
        client_process = Process(None, client)
        client_process.start()
        client_processes.append(client_process) """


if __name__ == "__main__":
    main()