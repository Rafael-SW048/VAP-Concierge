import os
from api.app_server import AppServer, serve as grpc_serve
from api.pipeline_repr import InferDiff

from api.common.enums import ResourceType

i = 0
APPS = ["hehehe"]
HOST="10.140.83.103"
APPSI = [5001]

# os.chdir("./app%d/workspace" %(i+1))
# os.system("ls")
# os.system("DATASET=%s yq -i '.dataset = env(DATASET)' configuration.yml" % (APPS[i]))
# os.system("hname=%s yq -i '.instances[1].hname |= env(hname)' configuration.yml" % (HOST+":"+str(APPSI[i])))