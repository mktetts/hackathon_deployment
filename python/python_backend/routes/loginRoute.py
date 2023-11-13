import sys
sys.dont_write_bytecode = True

from flask import Blueprint

from python_backend.contract.blockchain import isConnected
from python_backend.controllers.loginController import login
 

loginn = Blueprint('login', __name__)

@loginn.before_request
def checkBlockchain():
    isConnected()

loginn.route("/login", methods = ["POST"])(login)

   