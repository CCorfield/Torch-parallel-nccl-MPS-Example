# Using Torch, parallel, nccl, and MPS for multi-process, multi-GPU training

## Abstract
This document outlines how to do multi-process, multi-GPU neural-net training from within the Torch toolkit.

## Introduction
There are applications of machine learning where it is desirable to leverage the computing power of an ensemble of GPUs to train a neural net. The architecture of GPUs and CPUs allows us to consider parallel processing on the GPU and CPU. On the CPU this may take the form of parallel threads, or parallel processes, exploiting the multi-core architecture of contemporary CPUs, while on the GPU, this takes the form of computing kernels to break algebraic calculations into a swarm of parallel operations. In the Torch there are packages for both threaded and multi-process implementations. I will focus on the `parallel` package for multi-process training (for threaded approach see the `threads` package (GitHub location -- https://github.com/torch/threads) or `data parallel table` (GitHub  location -- https://github.com/torch/cunn/blob/master/doc/cunnmodules.md). 

## Torch-Parallel
The `parallel` package provides a framework for a parent process to fork multiple child processes and a means for these processes to communicate with each other (it uses ZeroMQ for the inter-process communication). You can find `parallel` on GitHub at https://github.com/clementfarabet/lua---parallel and you can install it with:

	$> luarocks install parallel

This package has a dependency on `libzmq` and `libzmq-dev` -- in my Ubuntu environment the installed versions are `libzmq3` and `libzmq3-dev`.

## nVidia-MPS
Now consider the following situation: multiple copies of a neural net resident on multiple GPUs, which may include several copies of the net on each GPU. You can imagine running separate training harnesses for each instance, training each net independently, and at some point saving each net and combining their parameters (e.g., take an average). Although this will work, it feels clumsy, and not quite in the spirit of parallel processing. Consider one of the problems with this approach: time slicing on a GPU. nVidia GPUs will happily support multiple processes and users sharing a GPU, but as nVidia points out, the processor will grant exclusive access to the client processes in a round-robin fashion, and since each process is unlikely to fully occupy all the cores of the GPU, it won’t be using all the processing power that is available. nVidia has partially addressed this issue with its “MPS” – Multi-Process Service – which allows computing requests from multiple processes run by the same user to be interleaved, leading to greater occupancy of the GPU. You can find nVidia's documentation on MPS at: https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf. 

To set up your GPUs to use MPS you can do the following:

Stop all the processes which are using the GPU. If you have Linux with X-Windows (and, presumptively, X uses one of the GPUs for display rendering), you can stop X-Windows with:

	$> sudo service lightdm stop

You may want to log in from a remote shell before stopping X, before you saw off the branch you are sitting on...

Set the GPUs to exclusive compute mode:
	
	$> sudo nvidia-smi -i <dev#> -c EXCLUSIVE_PROCESS

The reason for this is that a process called `nvidia-cuda-mps-server` will be the middleman brokering all computing requests into the GPU and only one `mps-server` at a time can serve in this capacity for a given GPU.
	
Now specify which GPUs are going to be part of the multi-process ensemble. In general you can partition your GPUs into disjoint groups, where each group can host a multi-process ensemble. To set up a group, do the following as root:

	$> export CUDA_VISIBLE_DEVICES=<dev#1>,<dev#2>,...
	
	$> nvidia-cuda-mps-control -d
	
This will launch a control daemon that will oversee the group of GPUs specified with `CUDA_VISIBLE_DEVICES`. The daemon launches and terminates instances of `mps-server` to broker computing requests to the GPUs within the ensemble. Since the GPUs are running in “exclusive compute mode”, you will only see one process running on them when you run `nvidia-smi`, and that process is `nvidia-cuda-mps-server`. Since only one user at a time has access to the group of GPUs, later users will block until the first user's processes have finished, at which point the MPS daemon will shut down the first user's instance of `mps-server` and start up a new instance of `mps-server` for the second user.

Once you have set up MPS and started the mps daemon (`nvidia-cuda-mps-control`) on a group of GPUs, you can run your `Torch` scripts on that group. However, you first need to set a couple of environment variables so that the CUDA libraries can find the instance of the MPS daemon that is managing the GPUs which you want to run on. You will need set the following environment variables:

	$> export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
	
	$> export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
	
In the above two lines, I have put in the default values that CUDA uses for its pipe and log directories. In practice, you will only need to set these environment variables when you have configured an MPS daemon to use non-default locations for its pipe and log directories, which will be the case if you have partitioned your GPUs into two, or more, groups. 

Now when you start up `Torch`, your `cutorch` package will have access to the GPUs which are in the managed group, and the device numbers will be 1, 2,… up to the number of GPUs in the group. In order to inform the CUDA library underneath `Torch` which GPUs to use you will need to set which devices are visible (the same as for the MPS daemon):

	$> export CUDA_VISIBLE_DEVICES=<dev1>,<dev2>,...

If you have set up MPS to control all your devices, you won't need to set this environment variable, the default behavior will take care of everything for you.	

To test your configuration, start `th` at a terminal prompt and entering the following:

	th> require "cutorch"
	
	th> cutorch.getDeviceCount()
	...
	
	th> cutorch.setDevice(<dev#>)
	...

	
If you have configured everything correctly, you won’t get any complaints. On the other hand, if MPS is not running properly you may find that commands such as `require 'cutorch'`, or `cutorch.setDevice(2)` will fail.

If you encounter problems, I suggest looking at the output from the daemon and server processes. In the log directory (default: `/var/log/nvidia-mps/`) there are two files:

	control.log
	
	server.log

Which provide time-stamped messages. Once, when I was setting up a new server, I was puzzled why I was not seeing `mps-server` listed in the output from `nvidia-smi`. There was a warning in the `server.log` file about the number of available file descriptors being insufficient. Sure enough, the new server had arrived configured with a limit of 1024 file descriptors -- you can use `ulimit -n` to see how many file descriptors you are allowed; if you need to increase the number, search around online for information on how to do it for your system.

Since it is often easiest to start from cookbook examples, I have uploaded some shell scripts which you can use as a starting point for configuring your own environment:

To start and stop an MPS environment for **_all_** the GPUs on your system, run these scripts as root:

	init_mps_for_all_gpus.sh
	
	stop_mps_for_all_gpus.sh [when you tear down the environment]
	
These two scripts use the default locations for the pipe and log directories, which means that you don’t have to set any user environment variables, the CUDA libraries incorporated within Torch will find these directories and communicate to the MPS daemon to start/stop `mps-server` processes. You may want to incorporate these scripts into your machine's boot sequence. 

If you do want to have each user set the locations of their pipe and log directories, have them source the following script:

	$>. set_mps_env_for_all_gpus.sh [yes, that is a period at the beginning of the line, short for 'source']

If you want to partition your GPUs into groups, run these scripts as root:

	init_mps_for_gpus.sh <dev1>,<dev2>,...
	
	stop_mps_for_gpus.sh <dev1>,<dev2>,... [when you tear down the environment]
	
Pass the group of GPUs into `init_mps_for_gpus.sh` as comma-separated list. This script will set up pipe and log directories in the following locations:

	/tmp/nvidia-mps_<dev1>_<dev2>_...
	
	/var/log/nvidia-mps_<dev1>_<dev2>

Note that nVidia counts devices from 0, whereas `cutorch` counts from 1. In this case, we use nVidia’s scheme. Since the pipe and log directories are no longer in their default locations, you must have every user set their locations in their environments before starting Torch scripts. You can have them source the following scripts:

	$>. set_mps_env_for_gpus.sh <dev1>,<dev2>,...
or	$>. set_mps_env.sh
	
The latter script will deduce the desired devices from the name of the MPS directory it finds in `/tmp`. Modify the scripts according to your own preferences.

One detail to note with MPS is that the daemon starts `mps-server` processes on demand, and that a given instance is always tied to one user, which prompts the question: “What happens if a second user tries to use the GPU(s) while the first user is running programs on them?” The short answer is that the second user’s application(s) will block until the first user’s last application quits. This is a consequence of the exclusive process compute mode. However, once all the first user’s processes have terminated, the MPS daemon will shut down the first user’s mps-server process and start up a new one with UID of the second user. Also note that when you run nvidia-smi you will not see your individual processes, but the `mps-server` process:

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

Since we're discussing software development, let me comment on what to do when things go wrong. At some point you will find that a script has not shut down properly, and you will need to manually purge running or wedged processes. I have found the following utilities useful (put in your .bashrc):

	`alias pslua='ps -ef | grep -v grep | grep luajit | cut -c -80'`
or	`alias pslua='ps -ef | grep -v grep | grep <username> | grep luajut | cut -c -80'`

	`function killpslua { for i in `pslua | awk '{print $2}'`; do kill -9 $i ; done; }`

You will also need to kill the `mps-server` instance if `nvidia-smi` shows it hanging on to GPU memory after you have shut down all your `Torch` processes, which indicates that a GPU compute-request has become wedged.  Simply `kill -9 <PID>` for the `PID` you see listed in the output of `nvidia-smi` (e.g., PID = 21388 in the above `nvidia-smi` output).

Once you have MPS running smoothly, it is time to tackle the next issue in parallel processing: transferring data between multiple processes.  

## nVidia-NCCL
A training process repeatedly pushes data through a net, calculates the derivatives of the loss function with respect to the net’s parameters, and then makes small adjustments to those parameters with the belief, or hope, that this will trace a path in parameter space to a location where the loss function is a minimum for any appropriate input data. Given that training data sets are large, it is desirable to devise a divide-and-conquer strategy which will be quicker than pushing all the training data through one instance of a net. One approach is to create multiple copies, or clones, of the net and to push different chunks of the data set through each copy which then raises the question: how do you keep the copies of the net in synch with each other? The two simplest methods are: (a) share accumulated gradient data before updating each net’s parameters; or, (b) update each net’s parameters with its own accumulated gradient data, and then share the updated parameters between all the nets. In the Torch environment each model’s parameters and gradient parameters can be stored as CudaTensors() which are resident in the memory of the GPUs and, therefore, the fastest way of transferring parameters or gradient parameters is to go directly from GPU to GPU and not take a detour through host memory. nVidia provides a mechanism for doing this called `nccl`, pronounced “nickel”. For Torch users, this means using CudaTensors, which may be resident on the same GPU or on different GPUs, and being able to copy data directly between them. nVidia makes `nccl` available as a C-library and header file (`nccl.h`), which means that its functionality can be exposed to Torch/Lua scripts – Lua provides a way to expose C APIs within Lua scripts (see `ffi` notes in http://luajit.org/ext_ffi.html). 

To make `nccl` available to your Torch environment, visit nVidia’s `nccl` page on GitHub (https://github.com/NVIDIA/nccl) and download the package of files to some directory, e.g., `/opt/nccl/`. Cd into that directory and follow the instructions to build the nccl library and run the optional tests. Install the library into your Torch environment with

	$> make PREFIX=<Torch install directory> install

For example, I have installed Torch under `/opt/torch`, which means the install directory is `/opt/torch/install` and after I have run nccl’s `make install`, copies of the nccl libraries (`libnccl.so*`) are in `/opt/torch/install/lib`.

What makes `nccl` useful is the following feature of Torch: If your training script includes the following line:

	params, gradParams = model:getParameters()

Then you will have persistent access to all of the *parameters* (`params`) and *gradient data* (`gradParams`) within your net (`model`), no matter how complex it is. These two tensors provide a very convenient way to access the net’s parameters. Indeed the following are equivalent:

	model:updateParameters(learningRate)
	
	params:add(-learningRate, gradParams)

If your model is resident on a GPU (courtesy a call to `model = model:cuda()`), then the two tensors `params` and `gradParams` are CudaTensors and resident on the same GPU, and you can now use `nccl` to transfer data directly between instances of them using functions such as `Bcast, `Reduce`, `AllReduce`, etc. 

## Example of transfering data using `parallel` and `nccl`
A simple example of combining multiple processes with `nccl` is provided by the following.

Scripts:

	parallel-nccl.lua
	nccl_ffi.lua

To run:

	$> th parallel-nccl.lua | sort

Since parallel processes execute asynchronously, I use `sort` to re-order their output so that it is easier to see what is going on.

* The example uses the `parallel` package to start a parent process and fork a number of child processes.

* The parent and children processes talk to each other, by first synchronizing (the parent calls `join` and the children call `yield`), and then passing data back and forth with the methods `send` and `receive`. Note that, courtesy Torch’s serialization, you can pass arbitrary data structures back and forth, so long as you have included the appropriate package (`require '<package>'`) which defines the object(s) that need to be serialized. Note that `parallel’s` communication channel is intended for host-to-host (or CPU-to-CPU) communication; having said that, you can use `parallel` to send a CudaTensor from one process to another and the serialization mechanism will create a CudaTensor() at the receiving end, although this would not be an efficient way to send parameters or gradient parameters of large neural nets.

* The processes set up a `nccl` communication channel. The parent creates a unique id to identify the group’s communication channel and passes that unique id to the child processes with `parallel’s` messaging facility. Each process (parent and children) then creates its own `nccl` communicator object. If you look at the `nccl` documentation you will find references to “ranks”. These are processes (or threads) by any other name. Conveniently, the `parallel` package and the `nccl` libraries use the same numbering scheme for their process-id’s (in `parallel’s` case) or ranks (in `nccl’s` case), so you might as well use the assigned `parallel.id` values as the “rank” numbers. 

* The section of script which sets up the `nccl` communicator object also illustrates how `Lua` scripts can interact with C-libraries. The relevant C API’s and data structures are provided by nVidia in `nccl.h`, which has been encapsulated in a `Lua-ffi` wrapper  `nccl_ffi.h`. The workings of `ffi` are beyond the scope of this discussion, so just treat `nccl_ffi.h` as another file to be “required”. 

* Once the communicator objects have been created, it is time to exercise the data transfer functionality. Each process creates send and receive buffers in the form of `CudaTensors`. Note that the allocated C-memory of a `CudaTensor` is accessed through `tensor:data()`, and thus any tensor should be 'contiguous' before using with `nccl`. We populate the tensor that is serving as the “sender” with some test data, and then call one of the `nccl` primitives. The processes will receive data in the tensor which is serving as the receive tensor. The awkward wording of the previous sentence is because the script also demonstrates how a single tensor can both send its data to other processes and then be populated by data it receives from those other processes. For simplicity, each child process populates its send tensor with its `parallel.id`, while the parent uses the arbitrray number `10`.

* Arguably the two most useful `nccl` functions are `AllReduce` and `Bcast`. `AllReduce` sums all the sending tensors and places the result into all the receiving tensors; the script implements two flavors: one where the sending tensor is also the receiving tensor (“in place”), and the second, where the receiving tensor is different from the sending tensor (“out of place”). Consider the case where we do an “in place” `AllReduce` on the `gradParams` tensor of each copy of a network – after the operation each of the `gradParams` will contain the sum of all the individual `gradParams`, which means they have the same values as would have been generated by passing all the training data through a single net (depending on circumstances, either leave the sum of the gradients alone, or divide by the number of nets to calculate an average). In other words, after `AllReduce` it suffices to call `model:updateParameters()` for each net. Alternatively, if you are considering ensemble training, you could first call `model:updateParameters()` on each net and then call `AllReduce` with the params tensors followed by `params:div(<#nets>)` to create an “average” of the nets.

You can use this script as a template for other `nccl` operations (such as, `Gather` and `Scatter`)

## Example of a multi-process, multi-GPU training harness
The following collection of scripts train and test a trained net using `parallel` and `nccl`.

Scripts:  

	Train.lua
	Test.lua
	Model.lua
	Network.lua
	DataSets.lua
	Evaluate.lua
	ccNesterov.lua
	nccl_ffi.lua

To train the net, run:

	$> th Train.lua

If this completes successfully, it will save a copy of the trained net in a `Torch` binary file `ExampleModel.t7`. The net is trained to recognize odd and even numbers in the range [1, 10] (yup, it is that simple). The net predicts odd or even by outputing a number between 0.0 and 1.0 -- if the number is < 0.5, the prediction is deemed to be 'even', and, similarly, if the output is > 0.5, the prediction is deemed to be 'odd'. A digression: I have included a version of Nesterov convergence which is more rigorous than the one in `Torch's` `optim` package. 

You can train the net in two different ways: 

- Share the gradient data between the models and then update each model independently

- Update each model using its own gradient data and then share the parameters (and average them)

Scroll down Network.lua to see the two implementations.

To test the net, run:

	$> th Test.lua

The tester loads the saved net, generates random numbers in the range [1,10], and passes them through the trained net to see which it thinks are odd and even. 

## Summary
This concludes a quick tour of `Torch`, `parallel`, `MPS`, and `nccl`. The intention is to get you up and running, without overwhelming you with extraneous details. If you are on a PC/Windows or Mac environment, and you can extend the above discussion to cover issues in those environments to help fellow Torch users, feel free to add your insights.


