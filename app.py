from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings

from ai import ai
from backend.auth import auth

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

app.register_blueprint(ai)
app.register_blueprint(auth)

if __name__ == "__main__":
    app.run(port="8000")
