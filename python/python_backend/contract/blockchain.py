import sys, os
sys.dont_write_bytecode = True

from web3 import Web3


from dotenv import load_dotenv
load_dotenv("../python_backend/config/ev.env")

w3 = None
network_id = None
connected = False
private_key = []
tx_count = 0


def isConnected():
    global w3, network_id, connected, tx_count
    try:
        # with open('../config/ev.env', 'r') as file:
        #     file_content = file.read()

        # print(file_content)
        # w3 = Web3(Web3.HTTPProvider(os.environ["BC_IP"]))
        current_directory = os.path.dirname(os.path.realpath(__file__))
        env_content_file = os.path.join(current_directory, '..', '..', 'envfile')
        with open(env_content_file, 'r') as file:
            ganache_ip = str(file.read())
        # w3 = Web3(Web3.HTTPProvider("http://ganache-cli-mktetts-dev.apps.sandbox-m3.1530.p1.openshiftapps.com"))
        w3 = Web3(Web3.HTTPProvider(ganache_ip))


        network_id = str(w3.eth.chain_id)
        connected = True
        return True
        # if True:
            
        # else:
        #     connected = False
        #     return False
    except Exception as exeption:
        connected = False
        return str(exeption)

# def compile():
#     with open("../../sc/contracts/userContract.sol") as file:
#         user_contract = file.read()
    
#     compiledUser = compile_standard(
#         {
#             "language" : "Solidity",
#             "sources"  : {
#                 "UserContract.sol" : { "content" : user_contract}
#             },
#             "settings" : {
#                 "outputSelection" : {
#                     "*" : { "*" : ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]}
#                 }
#             },
#         },
#         solc_version = "0.8.0",
#     )

# def unlockAccount(account, password):
#     try:
#         w3.geth.personal.unlock_account(account, password)
#         w3.geth.miner.start(1)
#         return True
#     except Exception as e:
#         return str(e)

# def lockAccount(account):
#     try:
#         w3.geth.personal.lock_account(account)
#         w3.geth.miner.stop()
#         return True
#     except Exception as e:
#         return str(e)

# def mine():
#     pending = w3.eth.filter("pending")
#     print(pending)
    
#     transactions = pending.get_new_entries()
#     print(transactions)

#     if len(transactions) > 0:
#         w3.geth.miner.start(1)
#     else:
#         w3.geth.miner.stop()
