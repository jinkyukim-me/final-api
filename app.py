from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings

# from ai import ai
from backend import root
from backend.auth import auth

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

app.config['MYSQL_HOST']='127.0.0.1'
app.config['MYSQL_USER']='test'
app.config['MYSQL_PASSWORD']=''
app.config['MYSQL_DB']='pythonlogin'

# app.register_blueprint(ai)
app.register_blueprint(auth)
app.register_blueprint(root)

if __name__ == "__main__":
    app.run(port="8000")
