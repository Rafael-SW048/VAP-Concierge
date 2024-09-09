from flask import Flask, jsonify

app = Flask(__name__)

experiment_status = 'running'
client_acknowledged = False

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": experiment_status,
        "acknowledged": client_acknowledged
    })

@app.route('/set_status/<status>', methods=['POST'])
def set_status(status):
    global experiment_status
    experiment_status = status
    return jsonify({"message": f"Experiment status set to {status}"}), 200

@app.route('/acknowledge', methods=['POST'])
def acknowledge():
    global client_acknowledged
    client_acknowledged = True
    return jsonify({"message": "Acknowledgment received"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6001)