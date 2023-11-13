import json, os, pandas as pd
import requests
from io import StringIO
from transformers import pipeline



w3 = None
network_id = None
connected = False
private_key = []
tx_count = 0

doctor_details_contract_address = None
prescription_details_contract_address = None
medicine_table = None
qa_pipeline = None


def deploy_contract(w3, cid):
    global doctor_details_contract_address, prescription_details_contract_address, medicine_table, qa_pipeline

    # cid = 'QmRTR2xVuxbVZ2umscMraQp4UsUwuCS2VHeJZNTKSi5zKL'
    ipfs_gateway_url = f'https://ipfs.io/ipfs/{cid}'
    response = requests.get(ipfs_gateway_url)
    content = response.text
    medicine_table = [str(content)] 

    # qa_pipeline = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")


    json_file_path = os.path.join(os.path.dirname(__file__), 'PrescriptionDetailContract.json')
    with open(json_file_path) as json_file:
        prescription_detail_contract_data = json.load(json_file)
        prescription_detail_contract_abi = prescription_detail_contract_data['abi']
        prescription_detail_contract_bytecode = prescription_detail_contract_data['bytecode']

    json_file_path = os.path.join(os.path.dirname(__file__), 'DoctorDetailContract.json')
    with open(json_file_path) as json_file:
        doctor_detail_contract_data = json.load(json_file)
        doctor_detail_contract_abi = doctor_detail_contract_data['abi']
        doctor_detail_contract_bytecode = doctor_detail_contract_data['bytecode']


    accounts = w3.eth.accounts
    sender = accounts[0]

    try:
        print("\n\n Contract Deployment Started...\n\n")

        contract = w3.eth.contract(abi=doctor_detail_contract_abi, bytecode=doctor_detail_contract_bytecode)
        tx_hash = contract.constructor().transact({'from': sender, 'gas': 6721975})
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print("DoctorDetailContract Deployed Successfully...")
        print(f'DoctorsDetails Contract Address : {tx_receipt["contractAddress"]}\n')

        doctor_details_contract_address = tx_receipt["contractAddress"]

        contract = w3.eth.contract(abi=prescription_detail_contract_abi, bytecode=prescription_detail_contract_bytecode)
        tx_hash = contract.constructor().transact({'from': sender, 'gas': 6721975})
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print("PrescriptionDetailContract Deployed Successfully...")
        print(f'Prescription Details Contract Address : {tx_receipt["contractAddress"]}\n')
        prescription_details_contract_address = tx_receipt["contractAddress"]

        return True

    except:
        print("Contract Deployment Failed")
        return False

