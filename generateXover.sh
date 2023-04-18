sat1=$1
sat2=$2
mkdir -p "./radsXover"
echo -e "${sat1} -> ${sat2} | $(date)"
radsxogen \
    -S "${sat1}" \
    -S "${sat2}" \
    --dual \
    --dt 0.5 \
    -V time,lat,lon,sla,sst,swh,wind_speed \
    "./radsXover/test.nc" \
    --log log.txt |& tee "status_${sat1}_${sat2}.txt"