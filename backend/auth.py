from flask import Blueprint, request, jsonify
# from utils import isLoggedin
from uuid import uuid4

auth = Blueprint('auth', __name__, url_prefix="/api/auth")

@auth.route('/register', methods=["POST"])
def register():
    pass

@auth.route('/login', methods=["POST"])
def login():
    # check email password
    token = uuid4()
    print(token)
    # uuid + user_id 페어를 DB에 저장
    return {
        'token': token
    }
