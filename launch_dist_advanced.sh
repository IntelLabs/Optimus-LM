#!/bin/bash

ppn=$1
num_nodes=$2
np=$((num_nodes * ppn))
PMI_SIZE=$np

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

# 6831496.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
JOB_ID="${PBS_JOBID%%.*}"

# Get the list of nodes from PBS_NODEFILE
NODES=()
while read -r NODE; do
    NODES+=("$NODE")
done < "$PBS_NODEFILE"

# Get hard node failures info
NODES_HNF=()
while read -r NODE; do
    if ! ping -c 1 -W 1 "$NODE" > /dev/null 2>&1; then
        NODES_HNF+=("$NODE")
    fi
done < "$PBS_NODEFILE"


# Get soft node failures info
NODES_SNF=()
launch_info_dir=/lus/flare/projects/Intel-Aurora/${USER}/launch_info/${JOB_ID}
mkdir -p "$launch_info_dir"
soft_node_failure_file_path="$launch_info_dir/failed_soft_nodes.txt"
if [[ -f "$soft_node_failure_file_path" ]]; then
    while read -r NODE; do
        NODES_SNF+=("$NODE")
    done < "$soft_node_failure_file_path"
fi

# Filter out bad nodes (soft and hard)
GOOD_NODES=()
while read -r NODE; do
    if ! grep -q "$NODE" <<< "${NODES_HNF[*]}" && ! grep -q "$NODE" <<< "${NODES_SNF[*]}"; then
        GOOD_NODES+=("$NODE")
    fi
done < "$PBS_NODEFILE"

# Check for sufficiency
if [[ ${#GOOD_NODES[@]} -lt $num_nodes ]]; then
    echo "Error: Not enough good nodes available. Required: $num_nodes, Available: ${#GOOD_NODES[@]}"
    exit 1
fi

# Create the host file
SELECTED_NODES=()
for ((i = 0; i < $num_nodes; i++)); do
    NODE="${GOOD_NODES[$i]}"
    SELECTED_NODES+=("$NODE")
done

echo "Alloted nodes path          : $PBS_NODEFILE"
echo "Soft node failure file path : $soft_node_failure_file_path"

echo "Number of allocated nodes   : ${#NODES[@]}"
echo "Number of hard failed nodes : ${#NODES_HNF[@]}"
echo "Number of soft failed nodes : ${#NODES_SNF[@]}"
echo "Number of good nodes        : ${#GOOD_NODES[@]}"
echo "Number of requested nodes   : $num_nodes"

# echo "Allocated nodes   : ${NODES[*]}"
echo "Failed hard nodes : ${NODES_HNF[*]}"
echo "Failed soft nodes : ${NODES_SNF[*]}"
# echo "Good nodes        : ${GOOD_NODES[*]}"
# echo "Selected nodes    : ${SELECTED_NODES[*]}"

# Creating host file
hostfile_path="$launch_info_dir/hostfile"
> "$hostfile_path"
for NODE in "${SELECTED_NODES[@]}"; do
    echo "$NODE" >> "$hostfile_path"
done

# Choosing master address
MASTER_ADDR=$(head -n 1 $hostfile_path | awk '{print $1}')

echo "Hostfile path  : $hostfile_path"
echo "Master address : $MASTER_ADDR"

# Final command to execute
cmd="MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=29500 PMI_SIZE=${PMI_SIZE}  mpiexec -hostfile $hostfile_path  -np ${np} -ppn ${ppn} $CPU_BIND ${exec_code}"

echo $cmd
eval $cmd

# Cleanup: kill any remaining python processes
# mpiexec -hostfile "$hostfile_path" -np ${num_nodes} -ppn 1 pkill python


# if [ $PMIX_RANK -eq 0 ]
# then
#   $*
# else
#   $* >& /dev/null
# fi