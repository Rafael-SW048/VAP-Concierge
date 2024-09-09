#!/bin/bash
# ./run.sh [MODE] [ALPHA] [ASCENT_COEF] [RESOLUTION]
# example: ./run.sh grad 0.3 1000 360
set -e

CE=/home/cc/ce
SRC=${CE}/src
MY_UID=$(id -ur)

mode=$1

# Allocator Config
export MONITOR_INT=10
export ALPHA=$2
export ASCENT_COEF=$3

# Pipeline Server Config
export DEFAULT_QUOTA=10000
export DEFAULT_RESOLUTION=$4

function cleanup {
	pkill -u $MY_UID python
}
trap cleanup EXIT INT TERM

function aggregate {
	video=$1
	ls  *.png | sort -n -k 2 -t- | xargs identify -format "%h\n" > resolution
	python $SRC/util_m/eval_acc.py -g ~/ce-res/gt-res/${video} -i $(pwd) > f1
	paste resolution f1 > tmp
	awk '{if ($2 != "") print NR, $1, $2}' tmp > $CE/$out_prefix-${video}.dat
}

if [[ $mode == "fair" ]] || [[ $mode == "gt" ]]; then
	export MONITOR_INT=1000000
	if [[ $mode == "gt" ]]; then
		export DEFAULT_RESOLUTION=-1
	fi
fi

out_prefix="${mode}"

if [[ $mode == "fair" ]]; then
	out_prefix="${out_prefix}-${DEFAULT_RESOLUTION}"
fi

if [[ $mode == "grad" ]]; then
	out_prefix="${out_prefix}-a${ALPHA}-t${ASCENT_COEF}-${DEFAULT_RESOLUTION}"
fi

out_dir="/home/cc/ce-res/${out_prefix}"
echo $out_dir
if (( $(ls /home/cc/ce/server-res/0 | wc -l) > 0 )); then
	rm ${CE}/server-res/0/* && rm ${CE}/server-res/1/*
fi

python allocator_m/allocator.py &

while [[ $(lsof -i -P -n | grep LISTEN | grep python | wc -l) < 2 ]]; do
	sleep 1
done

if [[ $mode == "gt" ]]; then
	python edge_m/edge.py -i 0 --app_id 0 -p 23333 --vpath ../edge-res/mix1-high-res.mp4 &
	python edge_m/edge.py -i 0 --app_id 1 -p 23334 --vpath ../edge-res/mix2-high-res.mp4
	pkill -U $MY_UID python
else
	python edge_m/edge.py -i 0 --app_id 0 -p 23333 --vpath ../edge-res/mix1.mp4 &
	python edge_m/edge.py -i 0 --app_id 1 -p 23334 --vpath ../edge-res/mix2.mp4
	pkill -U $MY_UID python
fi
 
if [[ $mode != "gt" ]]; then
	rsync -azP --exclude='*.avi' --exclude="**/*low*" $CE/server-res/ $out_dir
	cd $out_dir/0
	aggregate mix1
	cd $out_dir/1
	aggregate mix2
else
	rsync -azP $CE/server-res/0/*detection $out_dir/0
	rsync -azP $CE/server-res/1/*detection $out_dir/1

fi

