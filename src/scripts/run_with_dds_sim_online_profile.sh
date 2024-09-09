#!/bin/bash
set -e

video_1=$1
video_2=$2

EDGE_IP=("192.5.86.225" "192.5.86.216")
NUM_APP=2

cleanup() {
	sudo tc qdisc del dev eno1 ingress
	sudo killall -SIGINT python
}
trap cleanup SIGINT SIGTERM SIGKILL

copy_files() {
	dds_id=$1
	video_name=$2
	dds_tmpfs=/home/cc/tmpfs/dds${dds_id}/workspace/merged_results
	# copy inference_res
	rm -rf ${dds_tmpfs}/*
	cp /home/cc/data-set/${video_name}/merged_results/* ${dds_tmpfs}
	cd ${dds_tmpfs}
	rename "s/${video_name}/trafficcam_${dds_id}/" *

}

copy_files 1 ${video_1}
copy_files 2 ${video_2}

# start resource manager in background
cd /home/cc/vap-concierge/src
python -u api/concierge.py &
SERVER_IP=$!

sleep 5

# start server in background
for app_idx in $(seq $NUM_APP); do
	client_ip=${EDGE_IP[$(($app_idx - 1))]}
	if (( $app_idx < $NUM_APP )); then
		HOSTNAME=DDS${app_idx}\
			python -u sim/dds_sim.py -i ${app_idx} -c ${client_ip} & 
	else
		HOSTNAME=DDS${app_idx}\
			python -u sim/dds_sim.py -i ${app_idx} -c ${client_ip}
	fi
done
