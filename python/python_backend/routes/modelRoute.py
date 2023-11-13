import sys
sys.dont_write_bytecode = True

from flask import Blueprint

from python_backend.contract.blockchain import isConnected
from python_backend.controllers.modelController import predictWound, predictPrescription, questionAnswering, predictIndividualMedicine, speechToText

 

model = Blueprint('model', __name__)

@model.before_request
def checkBlockchain():
    isConnected()

model.route("/predictWound", methods = ["POST"])(predictWound)
model.route("/predictPrescription", methods = ["POST"])(predictPrescription)
model.route("/getMedicineForSymptoms", methods = ["POST"])(questionAnswering)
model.route("/predictIndividualPrescription", methods = ["POST"])(predictIndividualMedicine)
model.route("/speechToText", methods = ["POST"])(speechToText)





 