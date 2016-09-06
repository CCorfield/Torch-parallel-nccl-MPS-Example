# Torch-parallel-nccl-MPSExample

## Abstract:
This note outlines how to do multi-process, multi-GPU neural-net training from within the Torch toolkit.

## Introduction:
There are applications of machine learning where it is desirable to be able to leverage the computing power of an ensemble of GPUs to train a neural net. The architecture of GPUs and CPUs allows us to consider parallel processing on both the GPU and CPU. On the CPU this may take the form of parallel threads, or parallel processes, leveraging the multi-core architecture of contemporary CPUs. In the Torch there are packages for both threaded and multi-process. I will focus on the “parallel” package for multi-process training. 

## Torch-Parallel:
The parallel package provides a framework for a parent process to fork multiple child processes and a means for the processes to communicate with each other (it uses ZeroMQ for the inter-process communication). You can find parallel on GitHub at https://github.com/clementfarabet/lua---parallel and you can install it with:

```$> luarocks install parallel```

This package has a dependency on `libzmq` and `libzmq-dev` -- in my Ubuntu environment the packages are `libzmq3` and `libzmq3-dev`.

## nVidia-MPS:
Now consider the following situation: multiple copies of a neural net resident on multiple GPUs, which may include several copies of the net on each GPU. You can imagine running separate training harnesses for each instance, training each net independently, and at some point saving each net and combining them (e.g., take an average of the parameters). This feels clumsy and certainly not in the spirit of parallel processing. Consider one of the problems with this approach: time slicing on a GPU. nVidia GPUs will happily support multiple processes and users sharing a GPU, but as nVidia points out, the processor will grant exclusive access to the client processes in a round-robin fashion and since each process is unlikely to fully occupy all the cores of the GPU, it won’t be using all the processing power that is available. nVidia has partially addressed this issue with its “MPS” – Multi-Process Service – which allows computing requests from multiple processes run by the same user to be interleaved leading to greater occupancy of the GPU. To set up your GPUs to use MPS you can do the following:

1.	Stop all the processes which are using the GPU. If you are using a Linux box with X-Windows, you can stop it with:

	```$> sudo service lightdm stop```

You may want to log in from a remote shell before stopping X.

2.	Set the GPUs to exclusive compute mode. You may need to be root to do this:
	
	```$> sudo nvidia-smi -i <dev#> -c EXCLUSIVE_PROCESS```

	The reason for this is that a process called `mps-server` will be the middleman brokering all computing requests into the GPU and only mps-server at a time can serve in this capacity for a given GPU.
	
3.	Now specify which GPUs are going to be part of the multi-process ensemble. In general you can partition your GPUs into disjoint groups, where each group can run a multi-process training ensemble. To set up a group, do the following as root:

	```$> export CUDA_VISIBLE_DEVICES=<dev#1>,<dev#2>,...```
	```$> nvidia-cuda-mps-control -d```
	
This will launch a control daemon that will oversee the group of GPUs specified with `EXCLUSIVE_PROCESS`. The daemon launches and terminates instances of `mps-server` to broker computing requests to the GPUs within the ensemble. Since the GPUs are running in “exclusive compute mode”, you will only see one process running on them when you run `nvidia-smi`. Note that only one user at a time has access to the group of GPUs, later users will block until the first user has finished.

Once you have set up MPS and started the mps daemon (`nvidia-cuda-mps-control`) on a group of GPUs, you can run your Torch scripts on that group. However, you first need to set a couple of environment variables so that the CUDA libraries can find the instance of the mps daemon that is managing the GPUs you want to run on. You will need set the following environment variables:

	```$> export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps```
	```$> export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps```
	
In the above two lines I have put in the default values that CUDA uses for its pipe and log directories. In practice, you will only need to set these environment variables when you have configured an mps daemon to use non-default locations for its pipe and log directories, which will be the case if you have partitioned your GPUs into two, or more, groups. 

Now when you start up torch, your cutorch package will have access to the GPUs which are in the managed group, and the device numbers will be 1, 2,… up to the number of GPUs in the group. In order to inform the CUDA code underneath Torch which GPUs to use you will need to set which devices are visible (the same as for the MPS daemon):

	```$> export CUDA_VISIBLE_DEVICES=<dev1>,<dev2>,...```

If you have set up MPS to control all your devices, you won't need to set this environment variable, the default behavior will take care of everything for you.	

To test your configuration, start “th” at a terminal prompt and entering the following:

	```th> require 'cutorch'```
	```th> cutorch.getDeviceCount()```
	```...```
	```th> cutorch.setDevice(<dev#>)```
	```...```
	
If you have configured everything correctly, you won’t get any complaints. On the other hand, if MPS is not running you may find that commands such as `require 'cutorch'`, or `cutorch.setDevice(2)` will fail.

If you encounter problems, I suggest looking at the output from the daemon and server processes. In the log directory (default: `/var/log/nvidia-mps/`) there are two files:

	```control.log```
	```server.log```

Which provide time stamped messages. One time I was setting up a new server and I was puzzled why I was not seeing "mps-server" listed in the output from `nvidia-smi`, and there was a warning in the `server.log` about the number of available file descriptors. Sure enough, the new server had arrived configured with a limit of 1024 file descriptors.

Since it is often easiest to start from cookbook examples, I have uploaded examples shell scripts which you can use as a starting point for configuring your own environment:

1.	To start and stop an MPS environment for **_all_** the GPUs on your system, run these scripts as root:

	```init_mps_for_all_gpus.sh```
	```stop_mps_for_all_gpus.sh [later on, when you tear down the environment]```
	
These two scripts use the default locations for the pipe and log directories, which means that you don’t have to set any user environment variables, the CUDA libraries incorporated within Torch will find these directories and communicate to the MPS daemon to start/stop mps-server processes. You may want to incorporate these scripts into your start-up process. 

If you do want to have each user set the locations of the pipe and log directories, have them source the following script:

	```$>. set_mps_env_for_all_gpus.sh```

2.	If you want to partition your GPUs into groups, run these scripts as root:

	```init_mps_for_gpus.sh <dev1>,<dev2>,...```
	```stop_mps_for_gpus.sh [later on, when you tear down the environment]```
	
Pass the group of GPUs to init_mps_for_gpus.sh as comma-separated list. This script will set up pipe and log directories in the following locations:

	```/tmp/nvidia-mps_<dev1>_<dev2>_...```
	```/var/log/nvidia-mps_<dev1>_<dev2>```

Note that nVidia counts devices from 0, whereas cutorch counts starting from 1. In this case, use nVidia’s method. Since the pipe and log directories are no longer in their default locations you must have every user set their locations in their environments before starting Torch scripts. You can have them source the following scripts:
```
	$> set_mps_env_for_gpus.sh <dev1>,<dev2>,…
```
Pass the same list of devices as you used for setting up the MPS daemon.

One detail to note with MPS is that the daemon starts mps-server processes on demand, and that a given instance is always tied to one user, which prompts the question: “What happens if a second user tries to use the GPU(s) while the first user is running programs on them?” The short answer is that the second user’s application(s) will block until the first user’s last application quits. This is a consequence of the exclusive process compute mode. However, once all the first user’s processes have terminated, the MPS daemon will shut down the first user’s mps-server process and start up a new one with UID of the second user. Also note that when you run nvidia-smi you will not see your individual processes, but the mps-server process:
```
	$> nvidia-smi
	+-----------------------------------------------------------------------------+
	| NVIDIA-SMI 367.35                 Driver Version: 367.35                    |
	|-------------------------------+----------------------+----------------------+
	| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
	| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
	|===============================+======================+======================|
	|   0  GeForce GTX 970M    Off  | 0000:01:00.0     Off |                  N/A |
	| N/A   54C    P0    24W /  N/A |   1219MiB /  6069MiB |     23%   E. Process |
	+-------------------------------+----------------------+----------------------+
	|   1  GeForce GTX 970M    Off  | 0000:02:00.0     Off |                  N/A |
	| N/A   52C    P1    17W /  N/A |   1226MiB /  6077MiB |     61%   E. Process |
	+-------------------------------+----------------------+----------------------+

	+-----------------------------------------------------------------------------+
	| Processes:                                                       GPU Memory |
	|  GPU       PID  Type  Process name                               Usage      |
	|=============================================================================|
	|    0     21388    C   nvidia-cuda-mps-server                        1220MiB |
	|    1     21388    C   nvidia-cuda-mps-server                        1226MiB |
	+-----------------------------------------------------------------------------+
```
Once you have MPS running smoothly it is time to tackle the next issue in parallel processing: transferring data between multiple processes. nVidia’s documentation for MPS can be found at https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf 

## nVidia-NCCL:
A training process repeatedly pushes data through a net, calculates the derivatives of the loss function with respect to the net’s parameters, and then makes small adjustments to those parameters with the belief, or hope, that this will trace a path to a location in parameter space, where the loss function is vanishingly small for any conforming input data. Given that training data sets are large, it is desirable to devise a divide-and-conquer strategy which will be quicker than pushing all the data through one instance of a net. One approach that comes to mind is to create multiple copies, or clones, of the net and to push different chunks of the data set through each copy which then raises the question: how do you keep the copies in synch with each other? The two obvious approaches are either (a) share accumulated gradient data before updating each net’s parameters; or (b) update each net’s parameters with its own accumulated gradient data, and then share the updated parameters between all the nets. In the Torch environment each model’s parameters and gradient parameters are stored as CudaTensors() which are resident in the memory of the GPUs and, therefore, the fastest way of transferring parameters or gradient parameters is go directly from GPU to GPU and not take a detour through host memory. nVidia provides a mechanism for doing this called “nccl”, pronounced “nickel”. For Torch users, this means using CudaTensors, which may be resident on the same GPU or on different GPUs, and being able to copy data directly between them. nVidia makes “nccl” available as a C-library and header file, which means that its functionality can be exposed to Torch/Lua scripts – Lua provides a way to expose C APIs within Lua scripts. 

To make nccl available to your Torch environment, visit nVidia’s nccl page on GitHub (https://github.com/NVIDIA/nccl) and download the package of files to some directory, e.g., `/.../nccl/`. Cd into that directory and follow the instructions to build the nccl library and run the optional tests. Install the library into your Torch environment with

	```$> make PREFIX=<Torch install directory> install```

For example, I have Torch installed under `/opt/torch`, which means the install directory is `/opt/torch/install` and after I have run nccl’s `make install`, copies of the nccl libraries (`libnccl.so*`) are in `/opt/torch/install/lib`.

What makes nccl useful is the following feature of Torch: If your training script includes the following lines:

	```model = model:cuda()```
	```params, gradParams = model:getParameters()```

Then you will have persistent access to all of the parameters and gradient data within your net (`model`), no matter how complex it is. These two tensors provide a very convenient way to access the net’s parameters. Indeed the following are equivalent:

	```model:updateParameters(learningRate)```
	```params:add(-learningRate, gradParams)```

If your model is resident on a GPU (courtesy model:cuda()), then the two tensors params and gradParams are CudaTensors and resident on the same GPU, and can now use nccl to transfer data directly between instances of them using functions such as `Bcast, `Reduce`, `AllReduce`, et al. 

At this point, I suggest you take a look at the example script nccl-parallel.lua, which does the following:

1.	It uses `Torch-parallel` to start a parent process and fork a number of child processes
2.	These processes talk to each other, by synchronizing (the parent calls `join` and the children call `yield`), and then passing data back and forth with the methods `send` and `receive`. Note that courtesy Torch’s serialization, you can pass arbitrary data structures back and forth, so long as have included the appropriate package (`require '<package>'`) which define the object(s) that need to be serialized. Note that parallel’s communication channel is intended for host-to-host (or CPU-to-CPU) communication; having said that, you can send a CudaTensor from one process to another and the serialization mechanism will create a CudaTensor() at  the receiving end, but this would not be an efficient way to send parameters or gradient parameters of cloned neural nets.
3.	The processes set up a nccl communication channel. You will see where the parent creates a unique id to identify the group’s communication channel and passes that unique id to the child processes with parallel’s messaging functionality. Each process (parent and children) then creates its own communicator object. If you look at the nccl documentation you will find references to “ranks”. These are processes (or threads) by any other name. Very conveniently, the parallel package and the nccl libraries use the same numbering scheme for their process-id’s (in parallel’s case) or ranks (in nccl’s case), so you might as well use a “parallel.id” as the “rank” number. 
4.	The section of script which sets up the nccl communicator object also illustrates how Lau scripts can interact with C-libraries. The relevant C API’s and data structures are provided in `nccl.h`, which has been encapsulated in a Lua-ffi wrapper  `nccl_ffi.h`. The workings of ffi are beyond the scope of this discussion, so just treat `nccl_ffi.h` as another thing to be “required”. 
5.	Once the communicator objects have been set up, it is time to exercise the data transfer functionality. Each process creates send and receive buffers in the form of CudaTensors. We populate the tensor that is serving as the “sender” with some test data, and then call one of the nccl primitives. The processes will receive data in the tensor which is serving as the receive tensor. The awkward wording of the previous sentence is because the script also demonstrates how a single tensor can both send its data to other processes and then be populated by data it receives from those other processes. For simplicity, each child process populates its send tensor with its “parallel.id”, while the parent uses the arbitray number ‘10’.
6.	Arguably the two most useful nccl functions are `AllReduce` and `Bcast`. `AllReduce` sums all the sending tensors and places the result into all the receiving tensors; the script implements two flavors: one where the sending tensor is also the receiving tensor (“in place”), and the second, where the receiving tensor is different from the sending tensor (“out of place”). Consider the case where we do an “in place” `AllReduce` on the gradParams tensor of each copy of a network – after the operation each of the gradParams will contain the sum of all the individual gradParams, which means they have the same values as would have been generated by passing all the training data through a single net. In other words, after AllReduce it suffices to call model:updateParameters() for each net. Alternatively, if you are considering ensemble training, you could first call `model:updateParameters()` on each net and then call `AllReduce` with the params tensors followed by `params:div(<#nets>)` to create an “average” of the nets.
7.	There are other nccl operations (`Gather` and `Scatter`) which you can implement using this script as a template.

Example of a multi-process, multi-GPU training harness
Scripts:  
	```Train.lua, Test.lua, Model.lua, Network.lua, DataSets.lua, Evaluate.lua, ccNesterov.lua, and nccl_ffi.lua```
These scripts implement a simple training harness which illustrates how Torch, MPS, and nccl can be used to orchestrate a multi-process, multi-GPU training harness. 

To train the net, run:
	```$> th Train.lua```

To test the net, run:

	```$> th Test.lua```

You can try two training approaches, 


