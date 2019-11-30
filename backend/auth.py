from flask import Blueprint, request, jsonify
from utils import isLoggedin

auth = Blueprint('auth', __name__, url_prefix="/api/auth")

@auth.route('/register', method=["POST"])
def register():
    pass

def me():
    if isLoggedin(header)
