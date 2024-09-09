#!/bin/bash
# run this script with 
# cd vap-concierge/src && ./script/run_qyx.sh

EDGE_WORKDIR="/tmp/ramdisk/VAP-Concierge"
EDGE_PYTHONPATH="$EDGE_WORKDIR/src"

# Meta Parameters
NUM_APP=3

# The first one is empty, so that the index matches the app_idx
# APPS=("" "dds-adaptive" "dds-adaptive" "dds-adaptive")
# DATASETS=("" "uav-1" "coldwater" "roppongi")
# # "0" means false
# IS_FAKED=("" "1" "0" "0")

## MODIFY THIS
BASELINE_MODE=0 # 0: Concierge, 1: VideoStorm, 2: FairAlloc
INFERDIFF_MODE=true # if false then it will use accSen
READ_DURATION=4 # in second
DELTA_STEP=10 # step in determine_bandwidth()
## END OF MODIFY THIS

MIN_BW=80
# MAX_BW=1200
# PROFILING_DELTA=20
PROFILEING_DELTA_FRAMES=1
# MI=5

EXPERIMENT_PIDS=()
sudo pkill flask
sudo pkill python

cleanup() {
	for pid in "${EXPERIMENT_PIDS[@]}"; do
		echo "script -- INFO -- Killing process $pid"
		sudo kill -9 "$pid"
		sudo kill -2 "$pid"

		sudo pkill -f "python -u api/concierge.py"
		# TODO: ugly fix: for some reason multiple concierge servers are started
	done
	sudo kill -9 `sudo lsof -t -i:10001`
	sudo kill -9 `sudo lsof -t -i:10002`
	sudo kill -9 `sudo lsof -t -i:10003`
	exit 0
}
# run cleanup() on these signals
# TODO: it seems that this is not working for ctrl-C SIGINT
trap cleanup SIGINT SIGTERM SIGKILL EXIT

export PYTHONPATH=${EDGE_PYTHONPATH}
# I am using miniconda3 on my chameleon machine, set up conda for this script
source /home/cc/miniconda3/etc/profile.d/conda.sh
conda activate dds

# start gRPC Concierge server, also direct the output to log, run in bg
# once the client is checked in, the gRPC Pipeline client is initialized
# noted that MIN_BW cannot be 0. you cannot set the limit in `tc` to 0
# TODO: tee is not working
# concierge needs to know the mode
MIN_BW=$MIN_BW MAX_BW=$MAX_BW PROFILING_DELTA=$PROFILING_DELTA MI=$MI BASELINE_MODE=$BASELINE_MODE NUM_APP=$NUM_APP DELTA_STEP=$DELTA_STEP python -u api/concierge.py &
EXPERIMENT_PIDS+=($!)

echo "script -- INFO -- Starting Concierge server in process $!"

# wait for the establishment of listening of the Concierge server
while [[ $(sudo lsof -i -P -n | grep LISTEN | grep python -c) \
		-lt 1 ]]; do
	sleep 1
done

# dulplicate DDS Apps
# create an api to trigger apps to run (flask server is enough I guess)

# only for the server
for app_idx in $(seq $NUM_APP); do
	cd app || exit

	echo "script -- INFO -- Creating application ${app_idx} mirroring ${APPS[${app_idx}]}"
	rm -rf "app${app_idx}"
	echo ${APPS[$app_idx]}
	cp -frp ${APPS[$app_idx]} -T app"${app_idx}"

	cd - || exit
done

# start DDS servers in background
for app_idx in $(seq $NUM_APP); do
	data_port=$(( 10000 + app_idx ))
	cd app/app"${app_idx}" || exit
	rm -rf server_temp/*
	rm -rf server_temp-cropped/*

	echo "script -- INFO -- Starting DDS Server @ :${data_port}"
	APP_IDX=${app_idx} READ_DURATION=$READ_DURATION DELTA_STEP=$DELTA_STEP FLASK_APP=backend/backend.py flask run --port=${data_port} --host=0.0.0.0 > /tmp/null & 
	EXPERIMENT_PIDS+=($!)

	echo "script -- INFO -- Starting DDS Server in process $!"

	cd - || exit
done

# The information about the server and vap concierge should be embedded into a yml file
rm ./info.yml info.json
touch ./info.yml
num_app=$NUM_APP yq -i '.num_app = env(num_app)' info.yml
baseline_mode=$BASELINE_MODE yq -i '.baseline_mode = env(baseline_mode)' info.yml
profiling_delta=$PROFILING_DELTA yq -i '.profiling_delta = env(profiling_delta)' info.yml
max_bw=$MAX_BW yq -i '.max_bw = env(max_bw)' info.yml
inferdiff_mode=$INFERDIFF_MODE yq -i '.inferdiff_mode = env(inferdiff_mode)' info.yml
read_duration=$READ_DURATION yq -i '.read_duration = env(read_duration)' info.yml
delta_step=$DELTA_STEP yq -i '.delta_step = env(delta_step)' info.yml
for app_idx in $(seq $NUM_APP); do
	if (( 0 != $app_idx )); then
		export app_idx2=$app_idx
		dataset=${DATASETS[${app_idx}]} yq -i '.app[env(app_idx2)].dataset = env(dataset)' info.yml
		app=${APPS[${app_idx}]} yq -i '.app[env(app_idx2)].mode = env(app)' info.yml
		data_port=$(( 10000 + app_idx )) yq -i '.app[env(app_idx2)].data_port = env(data_port)' info.yml
		control_port=$(( 5000 + app_idx )) yq -i '.app[env(app_idx2)].control_port = env(control_port)' info.yml
		profile=${IS_FAKED[${app_idx}]} yq -i '.app[env(app_idx2)].is_faked = env(profile)' info.yml
	fi
done
IP_ADDRESS=$(ip route get 8.8.8.8 | awk -F"src " 'NR==1{split($2,a," ");print a[1]}')
server_ip=$IP_ADDRESS yq -i '.server_ip= env(server_ip)' info.yml
yq -o=json info.yml > info.json

# sending data and trigger the apps to run
curl -X POST -d @info.json --header "Content-Type: application/json" http://10.140.82.92:5030/

while true; do
	sleep 1
done

# # start DDS clients in background, the last one in foreground
# # the DDS client will also start the gRPC Pipeline server
# for app_idx in $(seq $NUM_APP); do
# 	control_port=$(( 5000 + app_idx ))

# 	cd app/app"${app_idx}"/workspace || exit

# 	# requires yq 4.x see https://github.com/mikefarah/yq/#install
# 	# set the server ip & port
# 	data_port=$(( 10000 + app_idx ))
# 	hname="127.0.0.1:$data_port" yq eval -i '.instances[1].hname |= env(hname)' configuration.yml
# 	# hname="10.140.83.103:$data_port" yq eval -i '.instances[1].hname |= env(hname)' configuration.yml

# 	echo "script -- INFO -- Starting DDS Client @ :${control_port}"
# 	# the last one run in foreground
# 	# the PUBLIC_IP 
# 	# which is the server_uri and client_uri inside AppInfo sent to the Concierge server
# 	# Pipeline clinet uses server_uri to establish channel to Pipeline server via PUBLIC_IP:CTRL_PRT
# 	# PipelineRepr uses client_uri to execute `tc` command on the client server
# 	dataset=${DATASETS[${app_idx}]} yq eval -i '.default.video_name |= env(dataset)' configuration.yml
# 	MAX_BW=$MAX_BW yq eval -i '.default.adaptive_test_display |= env(MAX_BW)' configuration.yml
# 	BASELINE_MODE=$BASELINE_MODE yq eval -i '.default.baseline_mode |= env(BASELINE_MODE)' configuration.yml

# 	PROFILEING_DELTA_FRAMES=$PROFILEING_DELTA_FRAMES \
# 	HOSTNAME=APP${app_idx}-${APPS[$app_idx]} CTRL_PRT=${control_port}  APP_IDX=${app_idx} \
# 		PUBLIC_IP="127.0.0.1" \
# 		python entrance.py &
# 		# PUBLIC_IP="127.0.0.1" \
# 	cd - || exit

# 	echo "script -- INFO -- Starting DDS Client in process $!"
# 	EXPERIMENT_PIDS+=($!)

# done

# while true; do
# 	sleep 1
# done
