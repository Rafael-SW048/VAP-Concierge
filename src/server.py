from flask import Flask, request, jsonify, send_file
import os
import json
import subprocess
import shutil
import signal

app = Flask(__name__)

@app.route('/start', methods=['GET'])
def index():
    global process
    process = 0
    print("GOT A START REQUEST!!!!")
    os.chdir('/tmp/ramdisk/VAP-Concierge/src/')
    process = subprocess.Popen(["bash", "runExperiment.sh"])   
    return "success"

@app.route('/stop', methods=["GET"])
def stop():
    print("GOT A STOP REQUEST!!!!")
    global process
    os.kill(process.pid, signal.SIGTERM)
    os.kill(process.pid, signal.SIGINT)
    # os.kill(process.pid, signal.SIGKILL)
    os.chdir('/tmp/ramdisk/VAP-Concierge/src/')
    return "Success"

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=6000)