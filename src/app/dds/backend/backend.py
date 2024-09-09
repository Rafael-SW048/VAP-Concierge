import os
import logging
from flask import Flask, request, jsonify
import yaml
from app.dds.backend.server import Server
from typing import Union

app = Flask(__name__)
server: Union[Server, None] = None


@app.route("/")
@app.route("/index")
def index():
    # TODO: Add debugging information to the page if needed
    return "Much to do!"


@app.route("/init", methods=["POST"])
def initialize_server():
    args = yaml.full_load(request.data)
    global server
    if not server:
        logging.basicConfig(
            format="%(name)s -- %(levelname)s -- %(lineno)s -- %(message)s",
            level="INFO")
        server = Server(args, args["nframes"])
        os.makedirs("server_temp", exist_ok=True)
        os.makedirs("server_temp-cropped", exist_ok=True)
        os.makedirs("server_profiling_temp", exist_ok=True)
        os.makedirs("server_profiling_temp-cropped", exist_ok=True)
        return "New Init"
    else:
        server.reset_state(int(args["nframes"]))
        return "Reset"


@app.route("/low", methods=["POST"])
def low_query():
    global server
    if server is not None:
        file_data = request.files["media"]
        results = server.perform_low_query(file_data, False)

        return jsonify(results)


@app.route("/high", methods=["POST"])
def high_query():
    global server
    if server is not None:
        file_data = request.files["media"]
        results = server.perform_high_query(file_data, False)
        return jsonify(results)


@app.route("/profiling-low", methods=["POST"])
def profiling_low_query():
    global server
    if server is not None:
        start_id = request.args.get("start_id")
        if start_id is not None:
            server.curr_profiling_fid = int(start_id)
            print(f"Profiling start_id {server.curr_profiling_fid}")
            file_data = request.files["media"]
            results = server.perform_low_query(file_data, True)
            return jsonify(results)


@app.route("/profiling-high", methods=["POST"])
def profiling_high_query():
    global server
    if server is not None:
        file_data = request.files["media"]
        results = server.perform_high_query(file_data, True)
        return jsonify(results)
