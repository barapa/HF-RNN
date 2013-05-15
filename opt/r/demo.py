"""
README.
==================


This is the demo file that puts all the pieces together.

It creates a data object (ie the problem to be solved) and a model; 
it wraps the model in functions that HF knows how to use. 

Then these functions, together with the initial vaule of the parameters
and the initial value of the damping, are given to HF.


Here is a suggested workflow. 
To make a new experiment, create a new file in the r/ folder 
(r stands for "runs"). For example, save a copy of this demo.py
to something like run1.py. Then you will make changes
to run1.py. Then you'd import run1.py and call run1.hf.optimize(),
where hf an instance the HF class instantiated below.

Upon instantiation, HF will create a folder for the log files.  It
will create a folder 'rnn_code/opt/r/demo/' (or
'rnn_code/opt/r/run1/') with two files: results and detailed_results.
Both files will have document the progress of the optimization as it
proceeds.  More details below.

To summarize:

1: create a data_object.

2: create and initialize model. The initialization, especially its
scale, is very important for RNNs when solving problems with long
range dependencies.

3: wrap the model in HF-friendly functions.

4: instantiate the HF object

================================================================

results, detailed results, and CG health
========================================

You may want to monitor the learning as it progresses. 
To do so, you should check out the logs.
But you could also visualize the hidden units of a batch by
Ctrl-C-ing the run, and calling the "visualize_batch" function 
below. 

[[SIDE NOTE: Visualizing the hidden units is helpful for finding a
sensible initialization.  If we initialize the RNNs parameters to be
too large (set the scale to 1.5 below), the hidden states will become
chaotic, which isn't useful: chaotic behaviour is so irregular that
generalization is impossible.

We want the hidden units to look like they're "doing something", that 
there is some activity, but that it's not off-the-charts saturated
and angry. 

On the other hand, if the initialization is such that the hidden units aren't
doing much, then the optimizer will also struggle to solve the problem, because
it won't have "anything to work with" at the early stages, which tends to result
in either failure to solve the pathological problems, or inferior solutions.

For real problems initialization may be less of a huge deal; but when
it comes to the pathological problems, we really want to have a hidden
state that does something interesting, which is affected, somewhat, by
events that occurred in the past, but at the same time which isn't too
worried by the current events, or else it will completely ignore the
past so that even HF couldn't recover. That's the intuition, roughly.

]]

Staring at detailed_results is instructive and useful. At each iteration
of HF, detailed_results prints the training and the test errors, the norms
of the parameter vectors and of the gradient (average gradient over grad_batches),
and a table that describes the run of CG. You should see a table with 9 columns,
whose meanings are the following.
- i: CG iter #
- model loss: the value of the quadratic model. Includes the regular and the
              structural damping.
- loss: the reduction in the \emph{objective}, i.e., loss(theta+x)-loss(theta)
        evaluated on the data specified by GN_batches.
- BT_cur: the value of the quadratic model where the curvature is evaluated
          at BT_batches.
- BT_best: the minimum of the BT_cur's so far
- |x|: the norm of the current parameter update
- |r|: the norm of the residual
- |x|_A: the A-norm of the parameter update
- sec/CG iter: how many seconds does a single iteration of CG take?

Model loss must always decrease due to the mathematics of CG. It can fail to
do so on crazy problems -- for example, on the addition problem with T=1000.
Once that happens, CG is helpless.

loss is the reduction of the objective evaluated at GN_batches. It is
the function model loss tries to approximate (sort of; note that model
loss uses a linear term evaluated on grad_batches, but the curvature
term is evaluated on GN_batches; however, loss is evaluated entirely
on GN_batches, which means that model loss doesn't entirely try to
approximate loss, unless grad_batches = GN_batches).

When things are healthy, loss should be negative and be similar to
model loss.  The reduction ratio, rho, which is (loss)/(model loss),
tries to maintain the health of the CG runs. When loss doesn't
decrease as much as model loss, it is a sign that the CG run isn't
very healthy. In this case, rho will be small, and the damping will
increase. Note that large damping (lambda from the paper) tends to fix
the health problem while weakening the second-order nature of the
algorithm, which is why lambad needs to be as small as possible, but not 
smaller.




Typical Behaviour
================

The pathological problems are challenging even for HF. Therefore, the
optimizer quickly starts predicting the mean of the labels, and then
it stays there for a while.  Perhaps for 10-20, even 30 HF
steps. Maybe longer. It can sometimes take 10,000 CG steps for the RNN
to start noticing --any-- long range dependencies with the It then
either "notices" the long range dependency and then starts making good
progress, or it never notices it. It takes several hours to reach to this
stage on my setup.

Sometimes a badly initialized RNN can make partial progress towards
solving the problem, in the sense that it has clearly noticed a long 
range dependency, but then it could get statr making very slow progress
towards a solution. 

Unfortunately these experiments take time. There is no way around it,
because a single forward pass is costly. However, there is a slim
chance that it's python which is slow. So if you implement the RNN in
C++ and find them to be much more efficient than my implementation,
please let me know at ilya@cs.utoronto.ca. 


The current code is different somewhat from the one in the paper,
is cleaner, and is easier to understand. This current version
works really well on the noiseless memorization problems. 

I have removed the L2 weight decay. Perhaps that's the culprit. 


Finally, it is much more convenient to turn python on
from the shell via the following command
% python -c 'import opt.r.demo as r; r.hf.optimize()'

rather than opening a terminal and running it from there. The latter
is much more tedious unless you are interested in inspecting the
hidden states of the RNN using the visualize function. But even then,
it may be more convenient to save the parameters by providing hf with
save_freq = 5, (for instance), do the learning without an interactive
session, but then load the parameters from an interactive session by
r.hf.load(); load will know where the file is saved.

"""

# 0: make sure the experiment is 100% reproducible
seed = 20
import gnumpy
gnumpy.seed_rand(seed)
import numpy
numpy.random.seed(seed)


# 1: choose a data object. To change the problem,
# simply replace pathological.add with anything else;
# e.g., pathology.mult or pathology.xor; choose whatever you want.
#
# All problems (other than xor) are initially set to T=200.  Feel free
# to edit opt.d.seq.pathology and create other problem variants. For
# example, try changing T. For example, change T, the batch size, or
# even invent a few problems on your own.
import opt.d.seq.pathological as pathology
data_object = pathology.mult


# 2: create a model. The current release only has an rnn.rnn object.
# Future releases will also have the mrnn, the lstm, as well
# as an interpreter for implementing RNNs that are good for HF;
# (which essentially is an interpreted theano). It is about 30% slower
# than the directly coded rnn, but it is much more convenient. 
import opt.m.rnn.rnn as rnn
import opt.utils.nonlin as nonlin
model = rnn.RNN(v = data_object.V, # the dimensions of the problem:
                o = data_object.O,
                out_nonlin = data_object.out_nonlin, # the output nonlinearity
                                                     # of the problem

                h = 100, # 100 hidden units
                hid_nonlin = nonlin.Tanh, # the nonlinearity of the hidden units

                struct_damp_nonlin = nonlin.Tanh, # use Tanh for "D'
                                                  # in the structural damping
                                                  # nonlinearity.

                )
# Initialize the MRNN.
model.initialize_self(num_in = 15,
                      scale = 1./15**0.5)  
# note the initial scale. The larger the weights, especially the
# hidden-hidden ones, the more active and aggressive the hidden state
# sequence is. If it's too timid, the RNN has nothing to work with,
# but the other extreme is bad also. I found 1.2 to work moderately
# well on all problems. However, I currently believe that much better
# results could be obtained with intelligently chosen initialization. 


# THE FOLLOWING IS A VERY IMPORTANT LINE OF CODE.
#
# If we want to solve problem with difficult long range dependencies,
# we can't initialize large input-to-hidden weights.  The large
# weights will overwhelm the hidden state and make it completely and
# have no dependence on the information it has seen in the past, so
# the vanishing gradients will be numerically zero. In contrast,
# smaller input-hidden weights don't change the hidden state that
# much, and the hidden state is at least somewhat aware of the
# existence of all the inputs in the past, giving HF something to work
# with.
#
# However, when the W_vh weights are too small, things tend to not
# work well.  Balance is essential.
model.W_vh *= 0.25


# weight decay. Consider experimenting with it. 
L2_decay = 2e-5


# wrap the model with an HF-friendly code:
class FunctionsForHFCls:
    def __init__(self):
        self.state=None
        self.old_bid = None


    def losses(self, batch, X):
        """
losses takes the model, computes the loss, and also computes the zero
one loss (or whatever it is you want). This is also the place where you 
add L2 regularization.
"""

        # get the loss we care about; OX are the predicitons.
        # This is useful if you want to do something else with them.
        ans, OX = model.unpack(X).loss(batch) 
        zero_one_loss = data_object.zero_one_loss(batch, OX) 

        batch_size = data_object.size(batch)
        L2_contribution = (X*X).sum()*(L2_decay*batch_size)*0.5


        return array([ans+L2_contribution, zero_one_loss])


    def grad_losses(self, batch, X):
        """
Since we already spend all this effort to compute the gradient,
we might as well also compute the loss. 
"""


        batch_size = data_object.size(batch)


        # OX are the network's predicitons, before being passed
        # through the network's output nonlinearity.
        grad, loss, OX = model.unpack(X).grad(batch)

        grad = grad.pack() + X * (L2_decay * batch_size)
    
        zero_one_loss = data_object.zero_one_loss(batch, OX)



        L2_contribution = (X*X).sum()*(L2_decay*batch_size)*0.5


        losses = array([loss+L2_contribution, zero_one_loss])

        return (grad, losses)
    
    def gauss_newton(self, bid, batch=None, X=None, R=None, damping_factor=None):
        """
The gauss_newton function is complicated because it does state caching. 
The Gauss Newton matrix is computed using R-prop and B-prop, both of which
require knowing the activites of the network's units. 
If we don't store these activites, we'd need to recompute them every time
we call gauss_newton. This is somewhat wasteful in situations where
we use the same minibatch. To overcome this problem, we simply store
(in self.state) the state of the network. Once HF is done with the 
current iteration, it tells us to forget (via bid=='forget') 
the state, to ensure that we recompute the state next time.

"""
        if bid == 'forget':
            self.state = None
            return

        model_X = model.unpack(X)
        model_R = model.unpack(R)

        # state caching: 
        if (bid != self.old_bid) or (self.state is None):
            self.state = model_X.forward_pass(batch)
            self.old_bid = bid

        # mu is the structural coefficient. 
        mu_times_lambda = float(damping_factor[1])


        ans = model_X.gauss_newton(batch,
                                   model_R, 
                                   state = self.state, 
                                   mu = mu_times_lambda).pack()
            
        batch_size = data_object.size(batch)
        L2_contribution = R * (L2_decay * batch_size)

        return ans + L2_contribution
functions_for_hf = FunctionsForHFCls() 





from opt.hf import HF
from numpy import array

hf = HF(path = __name__, 
        
        message = 'a demo run.',

        init_W = model,

        losses_fn = functions_for_hf.losses,
        grad_losses_fn = functions_for_hf.grad_losses,
        gauss_newton_fn = functions_for_hf.gauss_newton,

        data_object = data_object,


        ## This is the initial damping, and it is an important parameter.
        ## Basically, if a problem has many output units then we increase
        ## the damping, because the large number of the output units
        ## scale up the objective quite a bit. 
        ## The choice below is pretty sensible.
        init_damping = array([1, .03]) * data_object.num_outputs * 0.01,

        ## This is another choice worth trying. 
        ## init_damping = array([1, 0.03]) * data_object.T * 0.1,
        

        ## we do CG backtracking using the objective function rather
        ## than the qudaratic model on a heldout set. 
        BT_eval = 'objective',


        ## The upper limit on the number of CG steps per HF step.
        ## We use exactly 300 CG steps. 
        max_cg_iter = 300,
        min_cg_iter = 299,


        ## stop learning once the zero_one_loss goes below this point.
        ## Recall that losses is a vector whose first entry is the
        ## loss minimized by HF, and second entry is the zero one loss.
        ## See "def losses(..)" above.
        test_losses_threshold_fn = lambda losses: losses[1] < 0.001,

        
        ## save the parameters every so often. 
        save_freq=5)



        




def visualize_batch(batch=None, b=0):
    """
Will display an image with the input, hidden units, predictions,
and targets, in a single display. b is the element of the batch
that will be displayed.
"""
    if batch is None:
        batch = data_object(0)

    V, O, M = batch

    (V, H, OX) = state = model.unpack(hf.X).forward_pass(V)

    if len(H)==len(V)+1:
        H=H[1:]

    from pylab import imshow, gray, concatenate, array, zeros


    cim = lambda Z: array([bb.asarray()[b] for bb in Z]).T

    T = len(V)

    whitespace = zeros((1, T))+0.5

    ans = [whitespace,
           cim(V),
           whitespace,
           cim(H),
           whitespace,
           cim(OX),
           whitespace,
           cim(map(model.out_nonlin, OX)),
           whitespace,
           cim(O)]

    gray()
    imshow(concatenate(ans, 0), interpolation='nearest')


