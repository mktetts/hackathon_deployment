from python_backend.utils.success import Success
from python_backend.utils.error import Error
import python_backend.contract.blockchain
from python_backend.contract.admin import doctorDetailsInstance, prescriptionDetailsInstance, decodeInputData
from python_backend.contract.deploy import deploy_contract as deploy
from web3.middleware import geth_poa_middleware
from flask import jsonify, request, send_file
import sys, os
import qrcode
import numpy as np
from zxing import BarCodeReader
import cv2


sys.dont_write_bytecode = True

all_prescriptions = []

def add_prescription():
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

        prescriptionDetailContract = prescriptionDetailsInstance(w3)
        userPassword = values[2] + "@" + "12345"
        transaction_hash = prescriptionDetailContract.functions.addPrescription({
            "name": values[0],
            "email": values[1],
            "disease" : values[2],
            "prescription" : values[3],
            "comments" : values[4]
        }
        ).transact({
            "from": account
        })
        # print(transaction_hash.hex())
        success = True
        data_to_encode = transaction_hash.hex()

        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data_to_encode)
        qr.make(fit=True)

# Create an image from the QR Code instance
        img = qr.make_image(fill_color="black", back_color="white")
        
        img.save("qrcode.png")
    except Exception as e:
        print(e)
        return Error("Failed", str(e), 200)
    if success:
        current_directory = os.path.dirname(os.path.realpath(__file__))
        image = os.path.join(current_directory, '..', '..', 'qrcode.png')
        return send_file(image, mimetype='image/jpg')

        # return Success("Success", "Prescription Added Successfully", 200)
    return Error("Failed", "Prescription not Added", 200)

def get_all_prescription():
    all_prescriptions = []
    w3 = python_backend.contract.blockchain.w3
    # w3.middleware_onion.inject(geth_poa_middleware, layer=0)

    prescriptionDetailContract = prescriptionDetailsInstance(w3)

    doctorCount = prescriptionDetailContract.functions.prescriptionCount().call()
    for i in range(doctorCount):     
        all_prescriptions.append(prescriptionDetailContract.functions.prescription(i).call())

    return Success("Success", all_prescriptions, 200)


def get_prescription_details():
    values = list(request.json.values())
    w3 = python_backend.contract.blockchain.w3
    prescriptionDetailContract = prescriptionDetailsInstance(w3)

    doctorCount = prescriptionDetailContract.functions.prescriptionCount().call()
    for i in range(doctorCount):     
        all_prescriptions.append(prescriptionDetailContract.functions.prescription(i).call())
    for i in range(len(all_prescriptions)):
        if(values[0] == all_prescriptions[i][3]):
            result = {
                "PrescribedBy" : all_prescriptions[i][0],
                "DoctorsEmail" : all_prescriptions[i][1],
                "Disease" : all_prescriptions[i][2],
                "Prescription" : all_prescriptions[i][3],
                "Comments" : all_prescriptions[i][4], 
            }
            return Success("Success",result, 200)
    return Error("Failure", "Prescription Not found", 200)

def decodeQRCode():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    img = request.files['image']
    img.save("qrcode.png")
   # Read the QR code image
    current_directory = os.path.dirname(os.path.realpath(__file__))
    image = os.path.join(current_directory, '..', '..', 'qrcode.png')
    qr_code_image = cv2.imread(image)

    try:

        def decode_qr_code(image_path):
            reader = BarCodeReader()
            barcode = reader.decode(image_path)
            return barcode.parsed

        qr_code = decode_qr_code("qrcode.png")
        w3 = python_backend.contract.blockchain.w3
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        prescriptionDetailContract = prescriptionDetailsInstance(w3)
        res = decodeInputData(w3, qr_code, prescriptionDetailContract)
        result = {
            "hash" : qr_code,
            "data" : res
        }
        return Success("Success",result , 200)
    except:
        result = {
            "hash" : None
        }
        return Error("Failure",result , 200)