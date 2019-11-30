from flask import Blueprint, request, jsonify

root = Blueprint('root', __name__, url_prefix="/api")

@root.route('/me', methods=["GET"])
def me():
    token = request.headers.get('Authorization')
    print(token)
    print(token[6:])
    # DB에서 조회해서 돌려주기
    return {
        'name': 'me'
    }
