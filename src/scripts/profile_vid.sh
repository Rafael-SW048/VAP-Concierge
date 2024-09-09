#!/bin/bash
set -e

export DDS_ID=$1
WORK_DIR=/home/cc/dds${DDS_ID}
WORKSPACE=${WORK_DIR}/workspace
DATA_SET=${WORK_DIR}/dataset/trafficcam_${DDS_ID}/src
SRC=/home/cc/vap-concierge/src

VID_PATH=$2

gen_profile_bw_frame() {
	cd ${WORKSPACE}
	cat 0.* | sort -t , -nk2 -nk4 > $1
}

log() {
	msg=$1
	echo "$(date +%F_%T) ${msg}"
}

kill_server() {
	if [[ $(lsof -t -i:500${DDS_ID} | wc -l) -gt 0 ]]; then
		kill $(lsof -t -i:500${DDS_ID})
	fi
}

# kill_server
# [[ -e ${WORK_DIR} ]] || cp -r /home/cc/dds1 ${WORK_DIR}
# [[ -e ${WORK_DIR}/dataset/trafficcam_${DDS_ID} ]]\
# 	|| mkdir ${WORK_DIR}/dataset/trafficcam_${DDS_ID}
# [[ -e ${DATA_SET} ]] || mkdir ${DATA_SET}
# sed -i "s/trafficcam_1/trafficcam_${DDS_ID}/" ${WORKSPACE}/configuration.yml
# sed -i "s/127.0.0.1:5001/127.0.0.1:500${DDS_ID}/"\
# 	${WORKSPACE}/configuration.yml
# 
# rm -rf ${WORKSPACE}/0.* ${WORKSPACE}/1.* ${WORKSPACE}/results/*\
# 	${WORKSPACE}/../server_temp ${WORKSPACE}/../server_temp-cropped\
# 	${DATA_SET}/*
# cp ${VID_PATH}/*.png ${DATA_SET}

source /home/cc/anaconda3/etc/profile.d/conda.sh; conda activate dds
# cd ${WORK_DIR}
# PYTHONPATH=${WORK_DIR} FLASK_APP=backend/backend.py\
# 	flask run --port=500${DDS_ID} | tee server_log &
# server_pid=$!
# cd ${WORKSPACE}
# sleep 5
# PYTHONPATH=${WORK_DIR} python entrance.py
# kill_server

export ORIGIN=${VID_PATH}/results
export MERGED=${VID_PATH}/merged_results
# [[ -e ${ORIGIN} ]] || mkdir ${ORIGIN}
# [[ -e ${MERGED} ]] || mkdir ${MERGED}
# rm -rf ${WORKSPACE}/results/*cropped*
# cp ${WORKSPACE}/results/* ${ORIGIN}
# cd ${SRC}
# 
# export NFRAMES=$(ls ${VID_PATH}/*.png | wc -l)
# 
# log "Merging boxes in results..."
# python -u app/dds/scripts/merge_boxes.py
# log "Done!"
# 
# log "Collecting Bandwidth profile by frame..."
# gen_profile_bw_frame ${MERGED}/profile_bw_frame
# log "Done!"

log "Generating pareto fronts..."
cd ${SRC}
python -u app/dds/scripts/pareto_front.py | tee ${MERGED}/pf-10s
log "Done!"
