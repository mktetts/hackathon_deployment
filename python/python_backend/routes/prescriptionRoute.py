import sys
sys.dont_write_bytecode = True

from flask import Blueprint

from python_backend.contract.blockchain import isConnected
from python_backend.controllers.prescriptionController import get_all_prescription, add_prescription, get_prescription_details, decodeQRCode
 

prescription = Blueprint('prescription', __name__)

@prescription.before_request
def checkBlockchain():
    isConnected()

prescription.route("/addPrescription", methods = ["POST"])(add_prescription)
prescription.route("/getAllPrescriptions", methods = ["GET"])(get_all_prescription)
prescription.route("/getPrescriptionDetails", methods = ["Post"])(get_prescription_details)
prescription.route("/decodeQRCode", methods = ["Post"])(decodeQRCode)




 