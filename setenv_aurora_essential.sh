#!/bin/bash

# To avoid slow distributed initialization at large scale
# export CCL_KVS_MODE=mpi
export CCL_KVS_USE_MPI_RANKS=1

# To avoid seg faults
export ForceExtendedUSMBufferSize=256

# To avoid timeout issues with CCL KVS
export CCL_KVS_GET_TIMEOUT=600

# To avoid "mem handle cache limit is reached"
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=32768
export CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD=32768

# To avoid "scaleout_host_buf_size is not big enough to handle 1207959552 bytes"
export CCL_SYCL_SCALEOUT_HOST_BUF_SIZE=$((2 * 1024 * 1024 * 1024))

# To avoid "atl_mpi.cpp:911 comm_create: pmrt_kvs_get: error"
export CCL_ATL_TRANSPORT=mpi

# To avoid allgather inplace check.
export CCL_CHECK_INPLACE_ALIASING=0

# Setting offline mode for hugging face
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Debug related
export FI_LOG_LEVEL=warn
export FI_LOG_PROV=cxi
export MPIR_CVAR_PMI_VERSION=x
export MPIR_CVAR_REQUEST_ERR_FATAL=1

# To avoid OSError: AF_UNIX path too long
export TMPDIR=/tmp

# To avoid maxThreads warning
export NUMEXPR_NUM_THREADS=64