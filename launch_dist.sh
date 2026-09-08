#!/bin/bash

HOSTFILE_PATH=${HOSTFILE_PATH:-$PBS_NODEFILE}

ppn=$1
num_nodes=$2
np=$((num_nodes * ppn))
PMI_SIZE=$np

# Get the list of nodes from PBS_NODEFILE
NODES=()
while read -r NODE; do
    NODES+=("$NODE")
done < "$HOSTFILE_PATH"

# Choosing master address
MASTER_ADDR="${NODES[0]}"

# Constructing hosts argument
hosts=""
for ((i = 0; i < $num_nodes; i++)); do
    node="${NODES[$i]}"
    hosts="$hosts,${node:0:13}"
done
hosts="${hosts:1}"

# Bindings
CPU_BIND=-cpu-bind=list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96

# CCL affinity based on number ranks per node
if [[ "$ppn" == "12" ]]; then
    export CCL_WORKER_COUNT=1
    export CCL_WORKER_AFFINITY=5,13,21,29,37,45,57,65,73,81,89,97

    # export CCL_WORKER_COUNT=2
    # export CCL_WORKER_AFFINITY=5-6,13-14,21-22,29-30,37-38,45-46,57-58,65-66,73-74,81-82,89-90,97-98

    # export CCL_WORKER_COUNT=3
    # export CCL_WORKER_AFFINITY=5-7,13-15,21-23,29-31,37-39,45-47,57-59,65-67,73-75,81-83,89-91,97-99

    # export CCL_WORKER_COUNT=4
    # export CCL_WORKER_AFFINITY=5-8,13-16,21-24,29-32,37-40,45-48,57-60,65-68,73-76,81-84,89-92,97-100
    export PYTORCH_MPI_THREAD_AFFINITY=5,13,21,29,37,45,57,65,73,81,89,97
elif [[ "$ppn" == "8" ]]; then
    export CCL_WORKER_AFFINITY=5,13,29,37,57,65,81,89
    export PYTORCH_MPI_THREAD_AFFINITY=5,13,29,37,57,65,81,89
elif [[ "$ppn" == "6" ]]; then
    export CCL_WORKER_AFFINITY=5,21,37,57,73,89
    export PYTORCH_MPI_THREAD_AFFINITY=5,21,37,57,73,89
elif [[ "$ppn" == "4" ]]; then
    export CCL_WORKER_AFFINITY=5,13,57,65
    export PYTORCH_MPI_THREAD_AFFINITY=5,13,57,65
elif [[ "$ppn" == "2" ]]; then
    export CCL_WORKER_AFFINITY=5,57
    export PYTORCH_MPI_THREAD_AFFINITY=5,57
elif [[ "$ppn" == "1" ]]; then
    export CCL_WORKER_AFFINITY=5
    export PYTORCH_MPI_THREAD_AFFINITY=5
else
    echo "Warning Unsupported local size $ppn"
fi

# shift 2 # Arguments to python scripts starts from 3rd argument
shift 2
exec_code="$@"

# Final command to execute
cmd="MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=29500 PMI_SIZE=${PMI_SIZE} mpiexec -hosts $hosts -np ${np} -ppn ${ppn} $CPU_BIND ${exec_code}"

echo $cmd
eval $cmd
