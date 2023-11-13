import sys
sys.dont_write_bytecode = True

from flask import Blueprint

from python_backend.contract.blockchain import isConnected
from python_backend.controllers.adminController import add_doctor, deploy_contract, get_all_doctors
 

admin = Blueprint('admin', __name__)

@admin.before_request
def checkBlockchain():
    isConnected()

admin.route("/deployContract", methods = ["POST"])(deploy_contract)
admin.route("/addDoctor", methods = ["POST"])(add_doctor)
admin.route("/getAllDoctors", methods = ["GET"])(get_all_doctors)


 