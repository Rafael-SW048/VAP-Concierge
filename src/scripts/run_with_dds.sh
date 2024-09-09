#!/bin/bash
EDGE_IP=("192.5.86.225" "192.5.86.216")
EDGE_USER="cc"
EDGE_WORKDIR="/home/cc/vap-concierge"
EDGE_VPATH="$EDGE_WORKDIR/edge-res"
EDGE_PYTHONPATH="$EDGE_WORKDIR/src"
EDGE_CMD=(
	"export PYTHONPATH=${EDGE_PYTHONPATH};"
	"cd ${EDGE_PYTHONPATH}/app;"
	"python edge.py"
)

SERVER_IP="192.5.86.150"
NUM_APP=2

cleanup() {
	for app_idx in $(seq $NUM_APP); do
		ip=${EDGE_IP[$(($app_idx - 1))]}
		ssh $EDGE_USER@$ip 'pkill -U $USER python'
	done
	sudo killall python
}
trap cleanup SIGINT SIGTERM SIGKILL

killall python
sudo bash -c 'fuser -k 10001/tcp'; sudo bash -c 'fuser -k 10002/tcp'

# start resource manager in background
MAX_BW=200000 python -u api/concierge.py | tee log &

while [[ $(sudo lsof -i -P -n | grep LISTEN | grep python | wc -l) \
		< 1 ]]; do
	sleep 1
done

source /home/cc/anaconda3/etc/profile.d/conda.sh
conda activate dds
# start server in background
for app_idx in $(seq $NUM_APP); do
	data_port=$(( 10000 + $app_idx ))
	# client_ip=${EDGE_IP[$(($app_idx - 1))]}
	#sudo nvidia-docker run -p $control_port:$control_port -p $data_port:$data_port \
	#	--runtime=nvidia --gpus all --add-host=host.docker.internal:host-gateway \
	#	-e TF_FORCE_GPU_ALLOW_GROWTH="true" \
	#	1nfinity/pipeline-server:latest -i ${app_idx} -c ${client_ip} &
	cd app/dds/dds${app_idx}
	rm -rf server_profiling_temp/*
	rm -rf server_profiling_temp-cropped/*
	rm -rf server_temp/*
	rm -rf server_temp-cropped/*
	TF_FORCE_GPU_ALLOW_GROWTH="true"
		FLASK_APP=backend/backend.py flask run --port=${data_port} > /tmp/null & 
	cd -
done

# # wait for pipeline server to be initialized
sleep 5
echo 'Pipeline servers initialized'

# ssh to the edge and start streaming video
for app_idx in $(seq $NUM_APP); do
	control_port=$(( 5000 + $app_idx ))
	cd app/dds/dds${app_idx}/workspace
	# the last one run in foreground
	if (( $app_idx < $NUM_APP )); then
		DATA_DIR="/home/cc/vap-concierge/src/app/dds/dds${app_idx}/dataset/trafficcam_${app_idx}" \
			HOSTNAME=DDS${app_idx} CTRL_PRT=${control_port}  APP_IDX=${app_idx} \
			PUBLIC_IP="" \
			python entrance.py &
		cd -
	else
		DATA_DIR="/home/cc/vap-concierge/src/app/dds/dds${app_idx}/dataset/trafficcam_${app_idx}" \
			HOSTNAME=DDS${app_idx} CTRL_PRT=${control_port}  APP_IDX=${app_idx} \
			PUBLIC_IP="" \
			python entrance.py
	fi
done
