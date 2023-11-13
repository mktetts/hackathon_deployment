import json
import os
import python_backend.contract.blockchain

UserDict = {
	"userId" : "",
	"userType":"",
	"roleCode":"",
	"lastName":"",
	"firstName":"",
	"email":"",
	"password":"",
    "isActive":"",
	"createdBy":"",
	"createdDate":"",
	"changedBy":"",
	"changedDate":"",
	"companyCode":"",
	"key" : ""
}



def checkUser(UserRequest):
	# if UserRequest.get("key") != os.environ["KEY"]:
	# 	return "You are not Authorized"

	if len(UserDict) > len(UserRequest):
		field = str(set(UserDict) - set(UserRequest))
		return field + " field is not provided"
	elif len(UserDict) < len(UserRequest):
		field = str(set(UserRequest) - set(UserDict))
		return field + " is additionally provided field. Please remove"
	else:
		return True

def modify(userList):
	l = [""] * (len(UserDict))
	l[0] = userList[0]
	l[1] = userList[1]
	l[2] = userList[2]
	l[3] = userList[3]
	l[4] = userList[4]
	l[5] = userList[5]
	l[6] = userList[6]
	l[7] = userList[7]
	l[8] = userList[12]
	l[9] = userList[8]
	l[10] = userList[9]
	l[11] = userList[10]
	l[12] = userList[11]
	return l


def userInstance(w3):
	with open("../sc/build/contracts/UserContract.json") as contract:
		Contract = json.load(contract)
	user_contract_abi = Contract["abi"]
	user_contract_address = Contract["networks"][api.contract.blockchain.network_id]["address"]
	if user_contract_address == "":
		return False
	userContract = w3.eth.contract(address = user_contract_address, abi = user_contract_abi)
	return userContract

def decodeInputData(w3, transactionHash, contract):
	transaction = w3.eth.get_transaction(transactionHash)
	decoded_input = contract.decode_function_input(transaction["input"])
	print(decoded_input[1]["_user"])
	