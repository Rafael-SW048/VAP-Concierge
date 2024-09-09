import os
import logging
from flask import Flask, request, jsonify
from dds_utils import ServerConfig, writeResult
import json
import yaml
from .server import Server
from time import perf_counter as perf_counter_s
import csv

app = Flask(__name__)
server = None

from munch import *

def readAndUpdate(appNum, inferTime):
    data = []
    with open("./../inferTime-%d.csv" %(appNum), "r") as csv_file:
            time_reader = csv.reader(csv_file)
            time_data = list(time_reader)
            temp = time_data[-1]
            if(temp == []):
                temp = 0
            else:
                temp = float(temp[0])
            temp = inferTime
            time_data[-1] = [temp]
            data = time_data

    with open("./../inferTime-%d.csv" %(appNum), "w") as csv_file:
            time_writer = csv.writer(csv_file)
            time_writer.writerows(data)

@app.route("/")
@app.route("/index")
def index():
    # TODO: Add debugging information to the page if needed
    return "Much to do!"


@app.route("/init", methods=["POST"])
def initialize_server():
    args = yaml.load(request.data, Loader=yaml.Loader) 
    # yaml.SafeLoader will raise an Error when incoming qp/res are changed
    # temporary fix: change to yaml.Loader
    global iterator
    iterator = 0
    global appNum
    appNum = os.popen('pwd').read()
    appNum = int(appNum[appNum.rfind("app")+3])

    f = open("./../inferTime-%d.csv" %(appNum), "a+")
    f.write("\n")
    f.close()

    global server
    if not server:
        logging.basicConfig(
            format="%(name)s -- %(levelname)s -- %(lineno)s -- %(message)s",
            level="INFO")
        server = Server(args, args["nframes"])
        os.makedirs("server_temp", exist_ok=True)
        os.makedirs("server_temp-cropped", exist_ok=True)
        return "New Init"
    else:
        server.reset_state(int(args["nframes"]))
        return "Reset"


@app.route("/low", methods=["POST"])
def low_query():
    # count and update (for generalization of the chosen method), should it be in the form of csv? for easier reading
    file_data = request.files["media"]
    inferStart = perf_counter_s()
    start_frame = json.loads(request.form["json"])
    start_frame = int(start_frame["start_frame"])
    global server
    # global appNum
    if start_frame == -1: # normal inference
        results = server.perform_low_query(file_data)
    else: # profiling process
        results = server.perform_low_query(file_data, start_frame)
    inferTime = perf_counter_s() - inferStart
    # writeResult(appNum, inferTime, "lowQuery")

    # # readAndUpdate(appNum, inferTime)
    results["backendTime"] = inferTime

    return jsonify(results)


@app.route("/high", methods=["POST"])
def high_query():
    file_data = request.files["media"]
    inferStart = perf_counter_s()
    results = server.perform_high_query(file_data)
    inferTime = perf_counter_s() - inferStart
    # writeResult(appNum, inferTime, "highQuery")
    # readAndUpdate(appNum, inferTime)
    results["backendTime"] = inferTime

    return jsonify(results)
