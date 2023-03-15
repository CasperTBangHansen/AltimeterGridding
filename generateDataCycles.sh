#!/bin/bash

# Assign input arguments
sat=$1
passesStart=$(($2))
passesEnd=$(($3))
cycleStart=$(($4))
cycleEnd=$(($5))
nIter=$(($cycleEnd - $cycleStart))

mkdir -p "./radsCycles"
mkdir -p "./radsCycles/${sat}"
for (( cycle=$cycleStart; cycle<=$cycleEnd; cycle++ ))
do
    currentIter=$(($cycle - $cycleStart))
    echo -e "${sat} | ${currentIter}/${nIter} - $(date)"
    mkdir -p "./radsCycles/${sat}"
    rads2nc \
        -S "${sat}" \
        -C "${cycle}" \
        -P "${passesStart},${passesEnd}" \
        -V time,lat,lon,sla,sst,swh,wind_speed \
        --output "./radsCycles/${sat}/${sat}c${cycle}.nc" \
        --log log.txt
done