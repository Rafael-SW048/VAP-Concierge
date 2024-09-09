#!/bin/bash

# separate aws and dds results on old_videos

videos=("uav-1" "roppongi" "jakarta" "coldwater" "highway")
videos_size=${#videos[@]}
apps=("awstream-adaptive" "dds-adaptive")

mkdir old_videos_result

for video in ${videos[@]}; do
    for app in ${apps[@]}; do
        mkdir "./old_videos_result/$video-$app/"
        cp -rf "./app/$app/data-set/$video/"* "./old_videos_result/$video-$app/"
    done
done

# compress the result

tar cvzf old_videos_result.tar.gz old_videos_result/

# compress experiment results

cd ./app/
tar cvzf experiment_results.tar.gz baseline_* concierge_* old_results/ profiling_baseline/

cp ./experiment_results.tar.gz ..
cd ..

# send those files through scp
eval `ssh-agent -s`
ssh-add ~/mychameleonkey.pem

scp experiment_results.tar.gz old_videos_result.tar.gz cc@192.5.86.249:~/.


