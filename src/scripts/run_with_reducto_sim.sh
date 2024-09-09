#!/bin/bash
set -e

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
	sudo tc qdisc del dev eno1 ingress
	sudo killall python
}
trap cleanup SIGINT SIGTERM SIGKILL

for app_idx in $(seq $NUM_APP); do
	rm -f ../server-res/${app_idx}/*
done
rm -f /tmp/*detection
rm -f /tmp/*.png
rm -f sim/reducto1 sim/reducto2

# start resource manager in background
MAX_BW=10 python api/concierge.py &

while [[ $(sudo lsof -i -P -n | grep LISTEN | grep python | wc -l) \
		< 1 ]]; do
	sleep 1
done

# start server in background
for app_idx in $(seq $NUM_APP); do
	client_ip=${EDGE_IP[$(($app_idx - 1))]}
	if (( $app_idx < $NUM_APP )); then
		TF_FORCE_GPU_ALLOW_GROWTH="true" HOSTNAME=DDS${app_idx}\
			python sim/reducto_sim.py -i ${app_idx} -c ${client_ip} & 
	else
		TF_FORCE_GPU_ALLOW_GROWTH="true" HOSTNAME=DDS${app_idx}\
			python sim/reducto_sim.py -i ${app_idx} -c ${client_ip}
	fi
done

cleanup
