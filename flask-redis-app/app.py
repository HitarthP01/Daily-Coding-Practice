from flask import Flask, request, jsonify
import redis
from redis.exceptions import ConnectionError

app = Flask(__name__)
redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)

@app.route("/set", methods=["POST"])
def set_key():
    try:
        data = request.get_json()
        print(f"Received JSON: {data}")  # Debug log
        key = data.get("key")
        value = data.get("value")
        print(f"Setting key: {key}, value: {value}")  # Debug log
        if not key or not value:
            return jsonify({"error": "Key and value are required"}), 400
        redis_client.set(key, value)
        return jsonify({"message": f"Set {key} to {value}"}), 200
    except ConnectionError as e:
        return jsonify({"error": f"Redis connection failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get/<key>", methods=["GET"])
def get_key(key):
    try:
        value = redis_client.get(key)
        if value is None:
            return jsonify({"error": f"Key {key} not found"}), 404
        return jsonify({"key": key, "value": value}), 200
    except ConnectionError as e:
        return jsonify({"error": f"Redis connection failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/delete/<key>", methods=["DELETE"])
def delete_key(key):
    try:
        if redis_client.delete(key):
            return jsonify({"message": f"Deleted {key}"}), 200
        return jsonify({"error": f"Key {key} not found"}), 404
    except ConnectionError as e:
        return jsonify({"error": f"Redis connection failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# from flask import Flask, request, jsonify
# import redis
# from redis.exceptions import ConnectionError

# app = Flask(__name__)
# # Initialize Redis connection
# redis_client = redis.StrictRedis(host='redis', port=6379, decode_responses=True)
# @app.route('/set', methods=['POST'])
# def set_key():      
#     try:
#         data = request.get_json()
#         key = data.get('key')
#         value = data.get('value')
#         if not key or not value:
#             return jsonify({'error': 'Key and value are required'}), 400
#         redis_client.set(key, value)
#         return jsonify({'message': f"Key set {key} to value {value} successfully"}), 200
#     except ConnectionError as e:
#         return jsonify({'error': 'Redis connection error', 'details': str(e)}), 500
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True)
