#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: bash $0 <storage_dir>"
    exit 1
fi

# Set the storage directory path
STORAGE_DIR=$1

# Create actual directories
mkdir -p $STORAGE_DIR
mkdir -p $STORAGE_DIR/cached_models
mkdir -p $STORAGE_DIR/pretrain_parallel_models
mkdir -p $STORAGE_DIR/serial_models
mkdir -p $STORAGE_DIR/parallel_models
mkdir -p $STORAGE_DIR/job_logs
mkdir -p $STORAGE_DIR/launch_info

# Create symbolic links
ln -sfn $STORAGE_DIR/cached_models cached_models
ln -sfn $STORAGE_DIR/pretrain_parallel_models pretrain_parallel_models
ln -sfn $STORAGE_DIR/serial_models serial_models
ln -sfn $STORAGE_DIR/parallel_models parallel_models
ln -sfn $STORAGE_DIR/job_logs job_logs
ln -sfn $STORAGE_DIR/launch_info launch_info