import sys
sys.dont_write_bytecode = True

from flask import Blueprint

from python_backend.contract.blockchain import isConnected
from python_backend.controllers.userController import getAllUsers, addUser, getUserById, getUserHistory, getUserHistoryById, updateUser, deleteUser
 

user = Blueprint('user', __name__)

@user.before_request
def checkBlockchain():
    isConnected()

user.route("/getAllUsers", methods = ["GET"])(getAllUsers)
user.route("/getUserById/<string:id>", methods = ["GET"])(getUserById)
user.route("/addUser", methods = ["POST"])(addUser)
user.route("/updateUser/<string:id>", methods = ["PUT"])(updateUser)
user.route("/deleteUser/<string:id>", methods = ["DELETE"])(deleteUser)
user.route("/getUserHistoryById/<string:id>", methods = ["GET"])(getUserHistoryById)
user.route("/getUserHistory", methods = ["GET"])(getUserHistory)


 