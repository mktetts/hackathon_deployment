from python_backend.utils.success import Success
from python_backend.utils.error import Error
import python_backend.contract.blockchain
from python_backend.contract.admin import doctorDetailsInstance
from python_backend.contract.deploy import deploy_contract as deploy
from web3.middleware import geth_poa_middleware
from flask import jsonify, request
import sys
import datetime

sys.dont_write_bytecode = True

def deploy_contract():
    values = list(request.json.values())
    print(values)
    w3 = python_backend.contract.blockchain.w3
    status  = deploy(w3, cid=values[0])
    if status:
        return Success("Success", "Contract Deployed Successfully", 200)
    return Error("Failed", "Contract Deployment Failed", 200)

def add_doctor():
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
        userPassword = values[2] + "@" + "12345"
        transaction_hash = doctorDetailContract.functions.addDoctor({
            "name": values[0],
            "age": values[1],
            "email": values[2],
            "qualification": values[3],
            "hospital": values[4],
            "specialist": values[5],
            "phone" : values[6],
            "password" : userPassword
            

        }
        ).transact({
            "from": account
        })
        
        success = True
        now = datetime.datetime.now()

        # Define a time delta of 5 hours and 30 minutes
        time_difference = datetime.timedelta(hours=5, minutes=30)

        # Add the time difference to the current time to get IST
        ist_time = now + time_difference

        # Extract the current hour and minutes in IST
        current_hour_ist = ist_time.hour
        current_minute_ist = ist_time.minute

        phone_num = "+1234567890"  # Replace with the recipient's phone number
        message = f"Welcome To the Vino Pharmacy Shift.\n We are proud of you for working with Decentralized Pharmacy\n. Here is your Credentials:\n email : {values[2]} \n password : {userPassword}"

        # Send the message
        # kit.sendwhatmsg(phone_num, message, current_hour_ist, current_minute_ist + 1)


        all_doctors = []
        doctorCount = doctorDetailContract.functions.doctorCount().call()
        users = []
        for i in range(doctorCount):
            
            all_doctors.append(doctorDetailContract.functions.doctor(i).call())
        print(all_doctors)
    except Exception as e:
        print(e)
        return Error("Failed", str(e), 200)
    if success:
        return Success("Success", all_doctors, 200)
    return Error("Failed", "User not Added", 200)

def get_all_doctors():
    w3 = python_backend.contract.blockchain.w3
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)

    doctorDetailContract = doctorDetailsInstance(w3)
    all_doctors = []
    doctorCount = doctorDetailContract.functions.doctorCount().call()
    for i in range(doctorCount):     
        all_doctors.append(doctorDetailContract.functions.doctor(i).call())
    print(all_doctors)

    return Success("Success", all_doctors, 200)
