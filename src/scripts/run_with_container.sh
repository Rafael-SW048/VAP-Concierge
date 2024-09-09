#!/bin/bash
set -e

EDGE_IP=("127.0.0.1" "127.0.0.1")
EDGE_VID=("rene" "uav-1")
EDGE_USER="root"
EDGE_WORKDIR="/src/app/dds"
EDGE_PYTHONPATH="/src"
OBJECT_STORE_URI="https://chi.uc.chameleoncloud.org:7480/swift/v1/AUTH_0bf3e1ba5e8240f1b61c7d0f8fe3b29c/data-set"
NFRAMES=300
CONCIERGE_URI="192.5.86.150:5000"
NUM_APP=2

cleanup() {
	for app_idx in $(seq $NUM_APP); do
		ip=${EDGE_IP[$(($app_idx - 1))]}
		ssh -p ${app_idx}0022 ${EDGE_USER}@$ip pkill python
	done
	sudo killall python
	killall ssh
}
trap cleanup SIGINT SIGTERM SIGKILL

# start resource manager in background
MAX_BW=70000 python api/concierge.py &

while [[ $(sudo lsof -i -P -n | grep LISTEN | grep python | wc -l) \
		< 1 ]]; do
	sleep 1
done

# start server in background
for app_idx in $(seq $NUM_APP); do
	docker run --runtime=nvidia --rm\
		-p 1000${app_idx}:1000${app_idx} -t 1nfinity/dds-server\
		flask run --port=1000${app_idx} --host=0.0.0.0 &
done

# wait for pipeline server to be initialized
sleep 5
echo 'Pipeline servers initialized'

for app_idx in $(seq $NUM_APP); do
	echo DDS${app_idx} downloading frames from data set...

	i=$(( ${app_idx} - 1 ))
	vid_name=${EDGE_VID[${i}]}
	edge_ip=${EDGE_IP[${i}]}

	dataset_dir=${EDGE_WORKDIR}/dataset/trafficcam_${app_idx}
	dataset_src=${dataset_dir}/src

	download_frames_cmd=$(cat <<-EOF
		cd ${dataset_src};
		rm -f *.png;
		for fid in $(seq 0 $((${NFRAMES} - 1))); do
			frame_uri=${OBJECT_STORE_URI}/${vid_name}/\$(printf '%010d.png' \${fid});
			wget \${frame_uri};
		done;
	EOF
	)
	check_downloaded_cmd="ls ${dataset_src} | grep png | wc -l"
	frames_in_src=$(\
		ssh -p ${app_idx}0022 -o StrictHostKeyChecking=no\
		${EDGE_USER}@${edge_ip} "${check_downloaded_cmd}"
	)
	# if [[ ${frames_in_src} -ne ${NFRAMES} ]]; then
	# 	ssh -p ${app_idx}0022 -o StrictHostKeyChecking=no\
	# 	${EDGE_USER}@${edge_ip} ${download_frames_cmd}
	# fi
	echo Done!
done

for app_idx in $(seq $NUM_APP); do

	i=$(( ${app_idx} - 1 ))
	control_port=$(( 5000 + $app_idx ))
	vid_name=${EDGE_VID[${i}]}
	edge_ip=${EDGE_IP[${i}]}

	dataset_dir=${EDGE_WORKDIR}/dataset/trafficcam_${app_idx}
	dataset_src=${dataset_dir}/src
	profile_dir=${dataset_dir}/merged_results

	remote_cmd=$(cat <<-EOF
		. ~/.bashrc;
		[[ -e ${dataset_src} ]] || mkdir ${dataset_src};
		[[ -e ${profile_dir} ]] || mkdir ${profile_dir};
		rm -f ${profile_dir}/*;
		cp /data-set/${vid_name}/merged_results/* ${profile_dir}/;
		cd ${profile_dir};
		rename "s/${vid_name}/trafficcam_${app_idx}/" *;
		cd ${EDGE_WORKDIR}/workspace;
		PYTHONPATH=${EDGE_PYTHONPATH} PROFILE_DIR=${profile_dir}\
			HOSTNAME=DDS${app_idx} CTRL_PRT=${control_port}\
			APP_IDX=${app_idx} PUBLIC_IP=${edge_ip}\
			CONCIERGE_URI=${CONCIERGE_URI} GRPC_ENABLE_FORK_SUPPORT=false\
			python entrance.py;
	EOF
	)
	# the last one run in foreground
	if (( ${app_idx} < ${NUM_APP} )); then
		ssh -p ${app_idx}0022 -o StrictHostKeyChecking=no\
			${EDGE_USER}@${edge_ip} ${remote_cmd} &
	else
		ssh -p ${app_idx}0022 -o StrictHostKeyChecking=no\
			${EDGE_USER}@${edge_ip} ${remote_cmd}
	fi
done
