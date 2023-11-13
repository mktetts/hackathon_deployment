from python_backend.utils.success import Success
from python_backend.utils.error import Error
from python_backend.contract.user import checkUser, modify, userInstance
import python_backend.contract.blockchain
from web3.middleware import geth_poa_middleware
from flask import jsonify, request
import sys
import collections


sys.dont_write_bytecode = True


def question_answering():
    pass