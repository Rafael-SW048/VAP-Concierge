#!/bin/bash
# run this script with 
# cd vap-concierge/src && ./script/run_qyx.sh

EDGE_WORKDIR="/home/cc/vap-concierge"
EDGE_PYTHONPATH="$EDGE_WORKDIR/src"

# Meta Parameters
NUM_APP=3

# The first one is empty, so that the index matches the app_idx
APPS=("" "awstream-adaptive" "dds-adaptive" "dds-adaptive")
DATASETS=("" "uav-1" "coldwater" "roppongi")

BASELINE_MODE=true

MIN_BW=80
MAX_BW=1500
PROFILING_DELTA=80
MI=5
PROFILEING_DELTA_FRAMES=5

EXPERIMENT_PIDS=()

cleanup() {
	for pid in "${EXPERIMENT_PIDS[@]}"; do
		echo "script -- INFO -- Killing process $pid"
		kill -2 "$pid"
		kill -9 "$pid"

		# TODO: ugly fix: for some reason multiple concierge servers are started
		pkill -f "python -u api/concierge.py"
	done
	exit 0
}
# run cleanup() on these signals
# TODO: it seems that this is not working for ctrl-C SIGINT
trap cleanup SIGINT SIGTERM EXIT

export PYTHONPATH=${EDGE_PYTHONPATH}
# I am using miniconda3 on my chameleon machine, set up conda for this script
source /home/cc/miniconda3/etc/profile.d/conda.sh
conda activate dds-adaptive

# start gRPC Concierge server, also direct the output to log, run in bg
# once the client is checked in, the gRPC Pipeline client is initialized
# noted that MIN_BW cannot be 0. you cannot set the limit in `tc` to 0
# TODO: tee is not working
MIN_BW=$MIN_BW MAX_BW=$MAX_BW PROFILING_DELTA=$PROFILING_DELTA MI=$MI python -u api/concierge.py &

echo "script -- INFO -- Starting Concierge server in process $!"
EXPERIMENT_PIDS+=($!)

# wait for the establishment of listening of the Concierge server
while [[ $(sudo lsof -i -P -n | grep LISTEN | grep python -c) \
		-lt 1 ]]; do
	sleep 1
done

# dulplicate DDS Apps
for app_idx in $(seq $NUM_APP); do
	cd app || exit

	echo "script -- INFO -- Creating application ${app_idx} mirroring ${APPS[${app_idx}]}"
	rm -rf app"${app_idx}"
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
	APP_IDX=${app_idx} FLASK_APP=backend/backend.py flask run --port=${data_port} > /tmp/null & 

	echo "script -- INFO -- Starting DDS Server in process $!"
	EXPERIMENT_PIDS+=($!)

	cd - || exit
done

# start DDS clients in background, the last one in foreground
# the DDS client will also start the gRPC Pipeline server
for app_idx in $(seq $NUM_APP); do
	control_port=$(( 5000 + app_idx ))

	cd app/app"${app_idx}"/workspace || exit

	# requires yq 4.x see https://github.com/mikefarah/yq/#install
	# set the server ip & port
	data_port=$(( 10000 + app_idx ))
	hname="127.0.0.1:$data_port" yq eval -i '.instances[1].hname |= env(hname)' configuration.yml

	echo "script -- INFO -- Starting DDS Client @ :${control_port}"
	# the last one run in foreground
	# the PUBLIC_IP 
	# which is the server_uri and client_uri inside AppInfo sent to the Concierge server
	# Pipeline clinet uses server_uri to establish channel to Pipeline server via PUBLIC_IP:CTRL_PRT
	# PipelineRepr uses client_uri to execute `tc` command on the client server
	dataset=${DATASETS[${app_idx}]} yq eval -i '.default.video_name |= env(dataset)' configuration.yml
	MAX_BW=$MAX_BW yq eval -i '.default.adaptive_test_display |= env(MAX_BW)' configuration.yml
	BASELINE_MODE=$BASELINE_MODE yq eval -i '.default.baseline_mode |= env(BASELINE_MODE)' configuration.yml

	PROFILEING_DELTA_FRAMES=$PROFILEING_DELTA_FRAMES \
	HOSTNAME=APP${app_idx}-${APPS[$app_idx]} CTRL_PRT=${control_port}  APP_IDX=${app_idx} \
		PUBLIC_IP="127.0.0.1" \
		python entrance.py &
	cd - || exit

	echo "script -- INFO -- Starting DDS Client in process $!"
	EXPERIMENT_PIDS+=($!)

done

while true; do
	sleep 1
done
