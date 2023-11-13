from web3 import Web3
import json
# Connect to the Ganache CLI RPC server
w3 = Web3(Web3.HTTPProvider('http://ganache-cli-mktetts-dev.apps.sandbox-m3.1530.p1.openshiftapps.com'))

# Set your contract bytecode and ABI
with open('Counter.json') as json_file:
    contract_data = json.load(json_file)
    contract_abi = contract_data['abi']
    contract_bytecode = contract_data['bytecode']

def deploy_contract():
    accounts = w3.eth.accounts
    sender = accounts[0]

    # Deploy the contract
    contract = w3.eth.contract(abi=contract_abi, bytecode=contract_bytecode)
    print(contract)
    tx_hash = contract.constructor().transact({'from': sender, 'gas': 6721975})

    # Wait for the transaction to be mined
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    deployed_contract = w3.eth.contract(address="0xc3998BE1BEe2a8e85B090d5b6F67809049e70A9c", abi=contract_abi)
    print('Contract deployed at address:', deployed_contract.address)

    # Interact with the contract
    initial_count = deployed_contract.functions.count().call()
    print('Initial count:', initial_count)

    # Send a transaction to increment the count
    tx_hash = deployed_contract.functions.increment().transact({'from': sender})

    # Wait for the transaction to be mined
    w3.eth.wait_for_transaction_receipt(tx_hash)

    updated_count = deployed_contract.functions.count().call()
    print('Updated count:', updated_count)

deploy_contract()
