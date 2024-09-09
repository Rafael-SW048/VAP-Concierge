#!/bin/bash

MODE=$1

if [[ $MODE == "inferDiff" ]]; then
        echo "Switching mode to inferDiff........"
        mv api api-accSen
        mv api-inferDiff api
        echo "Done!!"
        # mv ./app/dds-adaptive ./app/dds-adaptive-accSen
        # mv ./app/dds-adaptive-inferDiff ./app/dds-adaptive
    elif [[ $MODE == "accSen" ]]; then
        echo "Switching mode to accSen........"
        mv api api-inferDiff
        mv api-accSen api
        echo "Done!!"
        # mv ./app/dds-adaptive ./app/dds-adaptive-inferDiff
        # mv ./app/dds-adaptive-accSen ./app/dds-adaptive
    fi
    