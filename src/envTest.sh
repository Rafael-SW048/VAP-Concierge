#!/bin/bash

# Do iteration for baseline first
AVAIL_DATASETS=("uav-1" "hochiminh" "jakarta" "jakarta-uav" "lagrange" "timesquare" "miami" "roppongi" "coldwater" "highway")
AVAIL_APPS=("awstream-adaptive" "dds-adaptive")
IS_FAKEDs=("1" "0" "0" "0" "0" "0" "0" "0" "0" "0" "0")

DATASETS_LENGTH=${#AVAIL_DATASETS[@]}

# MAX_BWs=(2400 2700 3300 6000)
MAX_BWs=(1500)
# MAX_BWs=(4500)
# MAX_BWs=(4500)
MIs=(5)
temp1=-1
PROFILING_DELTAs=(200)
# PROFILING_DELTAs=(10 30 50 100)
POS=0
pos_val=(2 7 8)

dataset="./combinations_cat.csv"
apps="./apps_new.csv"
is_faked="./id.csv"
DATA_IDX=()

readarray -t DATASETS_TOTAL < $dataset
readarray -t APPS_TOTAL < $apps
readarray -t IS_FAKED_TOTAL < $is_faked

LENGTH=${#DATASETS_TOTAL[@]} #(?)
LENGTH=1
ITERATION=1
# echo ${DATASETS_TOTAL[2]}
# echo ${APPS_TOTAL[2]}
# echo ${IS_FAKED_TOTAL[2]}
for batch in $(seq 0 $((ITERATION-1))); do
    for delta in ${PROFILING_DELTAs[@]}; do
        export PROFILING_DELTA=$delta
        for iter in $(seq 0 $((LENGTH-1))); do
            DATASETS=()
            APPS=()
            IS_FAKED=()
            DATASETS=(${DATASETS_TOTAL[$iter]})
            # APPS=("-" "dds-adaptive" "dds-adaptive" "dds-adaptive")
            # APPS=("-" "dds-adaptive" "dds-adaptive" "dds-adaptive")
            APPS=("-" "awstream-adaptive" "awstream-adaptive" "awstream-adaptive")
            IS_FAKED=("-" "0" "0" "0")
            export DATASETS=${DATASETS[@]}
            export APPS=${APPS[@]}
            export IS_FAKED=${IS_FAKED[@]}
            for MI_temp in "${MIs[@]}"; do
                export MI=$MI_temp
                for MAX_BW_temp in "${MAX_BWs[@]}"; do
                    # if [[ ${pos_val[@]} =~ $POS ]]; then
                    # if [[ $(echo "${pos_val[@]}" | fgrep -w "$POS") ]]; then
                    # if [[ $iter -lt 2 ]]; then MAX_BW_temp=900
                    # elif [[ $iter -lt 4 ]]; then MAX_BW_temp=1050
                    # elif [[ $iter -lt 6 ]]; then MAX_BW_temp=1200
                    # else MAX_BW_temp=1350; fi
                    export MAX_BW=$MAX_BW_temp
                    curl http://10.140.82.92:6000/start
                    sleep 20
                    source run_zharfanf.sh &
                    experimentPID=$!
                    echo "this run after executing"
                    echo $experimentPID
                    sleep 170
                    curl http://10.140.82.92:6000/stop
                    sudo kill -2 $experimentPID
                    sudo kill -9 $experimentPID
                    sudo kill -9 `sudo lsof -t -i:10001`
                    sudo kill -9 `sudo lsof -t -i:10002`
                    sudo kill -9 `sudo lsof -t -i:10003`
                    sleep 10
                        # echo $DATASETS - $MAX_BW
                        # echo $POS
                        # fi
                    POS=$((POS+1))
                done
            done
        done
    done
done


# eval `(ssh-agent -s)`
# ssh-add ~/mykey.pem
# path="/tmp/ramdisk/VAP-Concierge/src/app"
# ssh cc@192.5.86.177 "cd $path; ./move.sh dds-TA-randomCombs-inferDiff-rand-2"