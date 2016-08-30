
Example of a multi-gpu training harness, which illustrates the use of Torch/parallel to set up a multi-process
environment and nVidia/NCCL for inter-GPU communication of gradient parameters, so that multiple clones of a 
net can be trained simultaneously, pausing periodically to exchange accumulated gradient data, so that each
clone can update itself using all the gradient data from all the clones.

The equivalent of main() is in the file Train.lua, which defines the parent and child processes. The parent 
initializes various parameters, forks and execs the child processes and then passes them the parameters they
need. Once all the processes have been initialized, they embark on a training loop, with periodic data exchanges.
You can control how long the training lasts by setting the number of epochs, as well as early termination
if the modelś validation error rate gets beneath some validation threshold.
When it finishes the training harness will save copy of the model when it finishes and plot a graph of the 
loss function and validation error rates during training.

A few notes on the implementation:
0. This is written as a multi-process harness -- to illustrate how to do it -- it is presumed that you
   are already familiar with single process training, so I have not included any logic for
   "if(multiprocess) ... else ..."
1. We sync the processes via parallel.children:join() (called by parent process) and parallel.yield() 
   (called by child process)
2. We exchange data/messages using parallel.children:send(), parallel.children:receive(),
   parallel.parent:send(), and parallel.parent:receive().
3. We extract the parameters and gradient parameters into persistent one dimensional arrays "params" and 
   "gradParams". This is a technique which can be applied to complex networks, and greatly simplifies the
   training process.
4. We use nccl/AllReduce to exchange and sum the gradient data, so that each net ends up with a copy of the
   cumulative gradient data, which the net can use to update its model.
5. There are numerous ways of updating models. The most basic is SGD (Stochastic Gradient Descents), all the 
   rest are optimizations built on top of SGD. Most of these are implemented by the Torch/optim package; 
   however, I have included a homegrown version of Nesterov which is better than the Torch version.
6. The sample network provided is trained to learn odd and even numbers in the range 1-10. For simplicity,
   inputs are in the form of 1-hot vectors, meaning all zeros except for a single '1' at the obvious index.
   Training data is generated in the form of (trivial) minibatches with one member, but is written in
   a more general form.
7. The harness should be robust to different choices of numerical precision and whether you are using
   GPUs or not. It uses ":type()" or ":typeAs()" to allow for different choices of numerical precision.
8. The model used in this example is a (very) simple one which learns to identify odd and even integers
   in the range [1,10].

 
There is a Test.lua script which will load a saved model and run randomly generated testing data
through the model, so that you can see how accurate (or otherwise) your model is.