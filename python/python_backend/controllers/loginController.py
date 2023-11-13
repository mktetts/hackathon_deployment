from python_backend.utils.success import Success
from python_backend.utils.error import Error
import python_backend.contract.blockchain
from python_backend.contract.admin import doctorDetailsInstance
from python_backend.contract.deploy import deploy_contract as deploy
from web3.middleware import geth_poa_middleware
from flask import jsonify, request
import sys

sys.dont_write_bytecode = True

def login():
    status = python_backend.contract.blockchain.connected 
    success = False
    
    try:
        print(request.json)
        keys = list(request.json.keys())
        values = list(request.json.values())

        w3 = python_backend.contract.blockchain.w3
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        accounts = w3.eth.accounts

        account = accounts[0]

        doctorDetailContract = doctorDetailsInstance(w3)

        all_doctors = []
        doctorCount = doctorDetailContract.functions.doctorCount().call()
        doctor = []
        for i in range(doctorCount):
            
            doctor = doctorDetailContract.functions.doctor(i).call()
            if doctor[2] == values[0]:
                success = True
                break

        print(all_doctors)
    except Exception as e:
        print(e)
        return Error("Failed", str(e), 200)
    if success:
        return Success("Success", doctor, 200)
    return Error("Failed", "User not found", 200)