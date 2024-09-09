from flask import Flask, request, jsonify, send_file
import os
import json
import subprocess
import shutil
from time import sleep

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    print("GOT A REQUEST!!!!")
    # params: bandwidth limit and frame_id? how to send this json over http?
    # solution: using post method
    # video_name = request.args.get('video_name')
    infoApps = json.loads(request.data)
    # print(infoApps["num_app"])
    os.system("export PYTHONPATH='/tmp/ramdisk/VAP-Concierge/src/'")

    # extract params
    HOST=infoApps["server_ip"]
    PROFILING_DELTA=infoApps["profiling_delta"]
    NUM_APP=infoApps["num_app"]
    APPS=infoApps["app"]
    CLIENT_IP = request.headers.get('Host').split(":")[0]
    BASELINE_MODE=infoApps["baseline_mode"]

    os.chdir('/tmp/ramdisk/VAP-Concierge/src/')

    # duplicate apps
    for i in range(1, NUM_APP+1):
        if os.path.exists("./app/app%d" % (i)):
            print("removing dir app%d....." %(i))
            shutil.rmtree("./app/app%d" % (i))
            subprocess.run(["sudo", "rm -rf", f"/tmp/ramdisk/VAP-Concierge/src/app/app{i}"])
        os.system("echo duplicating app%d....." %(i))
        os.system("rsync -arv --exclude=data-set/* ./app/%s/ ./app/app%d/" % (APPS[i]["mode"], i))
        os.makedirs("./app/app%d/data-set/%s" %(i, APPS[i]["dataset"]), exist_ok=True)
        os.system("cp -rf ./app/%s/data-set/%s/profile/ ./app/%s/data-set/%s/results/ ./app/app%d/data-set/%s" % (APPS[i]["mode"], APPS[i]["dataset"], APPS[i]["mode"], APPS[i]["dataset"], i, APPS[i]["dataset"]))
        subprocess.Popen(["sudo", "/home/cc/miniconda3/envs/dds/bin/python", "cache_video.py"] + [f"{i}", APPS[i]["dataset"]])
        # subprocess to move images to the ram, so the dataset will be empty, do not wait
    sleep(8)

    # Starting app in the process
    os.chdir('/tmp/ramdisk/VAP-Concierge/src/')
    for i in range(1, NUM_APP+1):
        print("Starting app%d" %(i))
        os.chdir("./app/app%d/workspace" %(i))
        os.system("BASELINE_MODE=%s yq -i '.default.baseline_mode = env(BASELINE_MODE)' configuration.yml" % (BASELINE_MODE))
        os.system("DATASET=%s yq -i '.default.video_name = env(DATASET)' configuration.yml" % (APPS[i]["dataset"]))
        os.system("hname=%s yq -i '.instances[1].hname |= env(hname)' configuration.yml" % (HOST+":"+str(APPS[i]["data_port"])))
        os.system("PROFILEING_DELTA_FRAMES=%s \
	    HOSTNAME=APP%d-%s CTRL_PRT=%d  APP_IDX=%d \
		PUBLIC_IP=%s CLIENT_IP=%s CONCIERGE_URI=%s:5000 IS_FAKED=%s\
		python entrance.py &" % (PROFILING_DELTA, i, APPS[i]["mode"], APPS[i]["control_port"], i, HOST, CLIENT_IP, HOST, APPS[i]["is_faked"]))
        os.chdir("../../../")



    # with open('./profile-%s/profile-separated/bandwidthLimit.yml' %(video_name), 'w') as outfile:
    #     yaml.dump(bandwidth_dict, outfile, default_flow_style=False)
    
    # # Updating the configuration file
    # os.system("yq -i '.default.video_name = %s' configuration.yml" %(video_name))
    # # os.system("yq -i '.default.video_name = %s' configuration.yml" %(video_name))

    # Executing the program
    # subprocess.run(["python", "entrance.py"])

    return "success"

@app.route('/test', methods=["GET"])
def hello():
    print("GOT A REQUEST!!!!")
    return "Success"
    

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5030)