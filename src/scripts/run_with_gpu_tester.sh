#!/bin/bash
cleanup() {
	sudo killall python
}
trap cleanup SIGINT SIGTERM SIGKILL

[[ -z $NUM_APPS ]] 		&& NUM_APPS=1
[[ -z $ALLOCS ]] 		&& ALLOCS=100
[[ -z $EXP_LENGTH ]] 	&& EXP_LENGTH=120

IFS=: read -ra ALLOCS_ARR <<<"$ALLOCS"

killall python 2>/dev/null
sudo bash -c 'fuser -k 10001/tcp'; sudo bash -c 'fuser -k 10002/tcp'

# start resource manager in background
source /home/cc/miniconda3/etc/profile.d/conda.sh
conda activate torch
MAX_BW=200000 MI=100000 python -u api/concierge.py &

sleep 1

for i in $(seq 1 $NUM_APPS); do
	CUR_CGROUP_DIR="/sys/fs/cgroup/cpu/alnair${i}"
	[[ -d $CUR_CGROUP_DIR ]] || sudo mkdir $CUR_CGROUP_DIR
	CUR_ALLOC=${ALLOCS_ARR[$((i - 1))]}
	LD_PRELOAD=/home/cc/intercept-lib/build/lib/libcuinterpose.so\
		ALNAIR_VGPU_COMPUTE_PERCENTILE=$CUR_ALLOC CGROUP_DIR=$CUR_CGROUP_DIR\
		UTIL_LOG_PATH="sched_tester${i}_sm_util.log"\
		HOSTNAME="SCHED_TESTER${i}" CTRL_PRT=500${i} APP_IDX=${i}\
		python -u app/gpu_tester/scheduler_tester.py &
	echo "$!" | sudo tee $CUR_CGROUP_DIR/tasks 
done

sleep $EXP_LENGTH
cleanup
