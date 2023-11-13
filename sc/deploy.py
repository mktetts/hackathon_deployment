from web3 import Web3
import json
# Connect to the Ganache CLI RPC server
w3 = Web3(Web3.HTTPProvider('http://ganache-cli-mktetts-dev.apps.sandbox-m3.1530.p1.openshiftapps.com'))

# Set your contract bytecode and ABI
with open('build/contracts/Migrations.json') as json_file:
    migration_contract_data = json.load(json_file)
    migration_contract_abi = migration_contract_data['abi']
    migration_contract_bytecode = migration_contract_data['bytecode']


with open('build/contracts/PrescriptionDetailContract.json') as json_file:
    prescription_detail_contract_data = json.load(json_file)
    prescription_detail_contract_abi = prescription_detail_contract_data['abi']
    prescription_detail_contract_bytecode = prescription_detail_contract_data['bytecode']

with open('build/contracts/DoctorDetailContract.json') as json_file:
    doctor_detail_contract_data = json.load(json_file)
    doctor_detail_contract_abi = doctor_detail_contract_data['abi']
    doctor_detail_contract_bytecode = doctor_detail_contract_data['bytecode']


accounts = w3.eth.accounts
sender = accounts[0]

try:
    print("\n\n Contract Deployment Started...\n\n")
    contract = w3.eth.contract(abi=migration_contract_abi, bytecode=migration_contract_bytecode)
    tx_hash = contract.constructor().transact({'from': sender, 'gas': 6721975})
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print("Migrations Deployed Successfully...")
    print(f'Migrations Contract Address : {tx_receipt["contractAddress"]}\n')

    contract = w3.eth.contract(abi=doctor_detail_contract_abi, bytecode=doctor_detail_contract_bytecode)
    tx_hash = contract.constructor().transact({'from': sender, 'gas': 6721975})
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print("DoctorDetailContract Deployed Successfully...")
    print(f'DoctorsDetails Contract Address : {tx_receipt["contractAddress"]}\n')

    contract = w3.eth.contract(abi=prescription_detail_contract_abi, bytecode=prescription_detail_contract_bytecode)
    tx_hash = contract.constructor().transact({'from': sender, 'gas': 6721975})
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print("PrescriptionDetailContract Deployed Successfully...")
    print(f'Prescription Details Contract Address : {tx_receipt["contractAddress"]}\n')


except:
    print("Contract Deployment Failed")
    
