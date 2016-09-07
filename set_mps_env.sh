#!/bin/bash
# MPS Documentation: https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf
#
# Initialize the user environment after starting MPS
# We look for the pipe directory in /tmp/nvidia-mps* and deduce which devices are in play.
#
#set -x

# Message
echo "Setting MPS environment variables..."

# Look in /tmp for a pipe directory
suffix=''
devices=''
pipedir=`ls /tmp | grep 'nvidia-mps'`

if [ $pipedir != '' ]; then
  suffix=${pipedir//nvidia-mps?/}
  devices=${suffix//_/,}
fi

# Suffix is either a comma separated list of devices, or blank
if [ $suffix == '' ]; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
  export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
else
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps_$suffix
  export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps_$suffix
  export CUDA_VISIBLE_DEVICES=$devices
fi

#set +x
