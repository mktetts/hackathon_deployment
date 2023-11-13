import sys
sys.dont_write_bytecode = True

from flask import Flask
from flask_cors import CORS
import os
from python_backend.utils.error import Error


from python_backend.routes.adminRoute import admin
from python_backend.routes.loginRoute import loginn
from python_backend.routes.modelRoute import model
from python_backend.routes.prescriptionRoute import prescription





app = Flask(__name__)

app.config['JSON_SORT_KEYS'] = False

app.register_blueprint(admin, url_prefix = '/api')
app.register_blueprint(loginn, url_prefix = '/api')
app.register_blueprint(model, url_prefix = '/api')
app.register_blueprint(prescription, url_prefix = '/api')





CORS(app)

@app.errorhandler(404)
def handle_404(e):
    return Error("Failed", "URL not found", 404)

@app.route('/')
def index():
    return "Welcome to Flask"

if __name__ == '__main__':
    app.debug = True
    app.run()