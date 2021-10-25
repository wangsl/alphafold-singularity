#!/bin/bash

args=''
for i in "$@"; do 
    i="${i//\\/\\\\}"
    args="$args \"${i//\"/\\\"}\""
done

if [[ "$(hostname -s)" =~ ^g[r,v] ]]; then nv="--nv"; fi

if [ "$SLURM_TMPDIR" != "" ]; then
    bind="--bind $SLURM_TMPDIR:/tmp"
fi

singularity \
    exec $nv $bind \
    --bind /vast/work/public/alphafold:/alphafold-data:ro \
    /vast/wang/alphafold-20211025/alphfold.sif \
    /bin/bash -c "
source /opt/env.sh
$args
"

