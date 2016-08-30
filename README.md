# Torch-parallel-nccl-MPSExample
Example of multi-process, multi-GPU training using Torch-parallel, nVidia-nccl, and nVidia-MPS

The contents of this repository implement an example of how to  train a neural net using multiple processes and multiple GPUs from within the Torch toolkit. The components of this example comprise:
    shellscripts to manage nVidia's MPS daemons and servers
    Torch/Lua wrapper for nVidia's "nccl.h" header file, to make the NCCL library accessible to Torch/Lua scripts
    Torch/parallel scripts which implement a multiple-process, multi-GPU training and testing harness for a simple neural net
