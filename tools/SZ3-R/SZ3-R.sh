#!/bin/bash

COMMAND_FOLDER="./"
COMMAND="${COMMAND_FOLDER}/sz3"
LOG_FOLDER="./log"

DATA_FOLDER="/home/zyang/Desktop/datasets"

DATA_NAME_LIST=("density.d64" "pressure.d64" "velocityx.d64" "Uf48.f64.bin.dat" "sample_r_B_0.5_26.d64" "stat_planar.1.1000E-03.field.d64.1" "631-tst.bin.d64" "pressure_2000_1008x1008x352.d64.dat" "einspline_288_115_69_69.pre.d64.dat")
DIMENSIONS_LIST=("-3 384 384 256" "-3 384 384 256" "-3 384 384 256" "-3 500 500 100" "-1 33554433" "-3 500 500 500" "-1 102953248" "-3 352 1008 1008" "-4 69 69 115 288")

sz3_r(){
    for i in "${!DATA_NAME_LIST[@]}"; do
        DATA_PATH="${DATA_FOLDER}/${DATA_NAME_LIST[i]}"
        DATA_OUT_PATH="${DATA_PATH}.sz.out"
        CONFIG_PATH="${COMMAND_FOLDER}/$3"
        PARAMS="$1 -i ${DATA_PATH} -o ${DATA_OUT_PATH} ${DIMENSIONS_LIST[i]} -c ${CONFIG_PATH} -M REL $2 -a $5"

        echo "Running command: $COMMAND $PARAMS"
        OUTPUT=$($COMMAND $PARAMS 2>&1)

        if [ $? -eq 0 ]; then
            echo "Command executed successfully."
            echo "Output:"
            echo "$OUTPUT"
        else
            echo "Command failed with the following output:"
            echo "$OUTPUT"
        fi
        LOGFILE="${LOG_FOLDER}/$4/${DATA_NAME_LIST[i]}.log"
        echo "$OUTPUT" > $LOGFILE
        echo "Output saved to $LOGFILE"
    done
}

sz3_r "-d" "1e-2" "sz3.config" "sz3_delta_cubic.log" "-e 9 65536e-9 16384e-9 4096e-9 1024e-9 256e-9 64e-9 16e-9 4e-9 1e-9"
