"""
This is the main file that defines the HF optimizer. 

In short, this file defines a massive class called HF. It has a ton of
parameters, but luckily most of them have sensible default values.

The details are described below. First we define a few helpful utility
functions, then we go and code the class itself. This code isn't
necessarily the most beautiful but it works.

"""

import numpy as np
import gnumpy as g

def ip(x,y): return (x*y).sum()
def norm2(x): return ip(x,x)
def norm(x): return float(np.sqrt(norm2(x)))
def normalize(x, M_diag=None):
    if M_diag is None:
        n = norm(x)
    else:
        n = float(np.sqrt(ip(x,M_diag*x)))
    return n, x/n

def progress_heuristic(losses):
    """
The progress heuristic: how to determine that it's time to stop CG?
There are many possible ways to address this question, and the
progerss heuristic is a pretty decent way; we look at the sequence of
losses as defined by the quadratic model. That is, losses[i] =
ip(g,x_i) -.5*ip(x_i,Ax_i).  If the progress heuristic notices that
losses stops getting better relative to the size of the reduction,
it'll return True, which means that CG should stop. Otherwise it
should return false. It is copied verbatim from the original HF paper.
"""
    eps = 0.0005

    i = len(losses)
    k = int(max(10, 0.1*i))

    if len(losses) < k+1:
        return False

    phi_x_i   = losses[-1]
    phi_x_imk = losses[-1-k]

    if i>k and phi_x_i<0 and (phi_x_i-phi_x_imk)/phi_x_i < k*eps:
        return True
    else:
        return False

def damping_heuristic(damping, rho):
    """
The damping heuristic implements the Levenberg-Marquardt 
adaptive rule for lambda; it takes a lambda and a rho, and returns
a new lambda. If rho is small, lambda gets larger, and if rho is large,
lambda gets smaller. Also copied verbatim from the original HF paper.

"""

    from numpy import isnan
    decr  = 2./3
    incr = 1./decr

    if rho < 1/4. or isnan(rho):    # the reductino was bad, rho was small
        d = incr
    elif rho > 3/4.:                # the reduction was good since rho was large
        d = decr
    else: 
        d = 1.
    return damping*d


class HF(object):
    """
So this is the HF class. It implements something very similar to the
original HF algorithm, as described in "Deep Learning via Hessian Free
Optimization" by James Martens with very small differences.
    """

    def __init__(self,
                 # path: we give HF a path, where it will store the logs.
                 # HF doesn't print things on the screen since it has 
                 # too much stuff to print.
                 path, 


                 # a message you can leave yourself, which will be printed
                 # regularly into the logs. 
                 message,


                 init_W,
                 # the initial model with initial parameters. Don't
                 # mess it up, as it's really important to get it
                 # right, especially for RNNs trained on the
                 # pathological problems. Ordinary problems are more
                 # forgiving.


                 # the main core functions defining our model all of
                 # them accept a parameter vector and a
                 # minibatch. They return losses and or other
                 # parameter vectors.
                 losses_fn, 
                 grad_losses_fn, 
                 gauss_newton_fn,


                 # the data_object object. Its interface is discussed
                 # in detail in opt/d/seq/pathological.py
                 data_object, 

                 
                 # the initial value of the damping. It can be a
                 # vector.  It is sensible to use a vector
                 # init_damping whenever we use structural damping.
                 init_damping,

                 
                 # the batches used for evaluating the gradient.
                 # 
                 # Specifically, grad = sum(grad(data_object(i)) for i in grad_batches) / 
                 #                      sum(size(data_object(i)) for i in grad_batches) / 
                 #
                 # where size is actually a function of data_object
                 # that returns the size of the minibatch. In the
                 # above, grad implicitly takes a parameter. If grad_batches is None,
                 # then grad_batches will default to data_object.train_batches
                 grad_batches=None,


                 # The batches used for evaluating the curvature
                 # matrix, B.  In other words, we'd call the
                 # gauss_newton function on each minibatch specified
                 # by GN_batches.
                 #
                 # If you choose to set GN_batches yourself, it is
                 # very important to make sure that GN_batches is a
                 # subset of grad_batches.
                 #
                 # Things can work even if this is violated, but
                 # performance will suffer.
                 #
                 #
                 # Furthermore, in this implementation, GN_batches is
                 # used as an early stopping criterion. That is, if
                 # loss(GN_batches) starts increasing, then the run of
                 # CG is terminated.  We found this technique to be
                 # helpful, and will discuss it in a future paper on
                 # HF. However we don't use this feature in the
                 # experiments provided by this code.
                 GN_batches=[0],


                 # BT_batches is used for backtracking. See the
                 # original HF paper.  It provides a different set of
                 # data for evaluating the curvature matrix.
                 # 
                 # There are many ways of evaluating the quality of an
                 # intermediate solution. The standard way is the
                 # following: First say the gradient g is computed
                 # with grad_batches.
                 #
                 # Say that GN_batches define a curvature matrix B, in
                 # the sense that we can multiply by B using the
                 # gauss_newton function and the list GN_batches.
                 #
                 # Then BT_batches define a curvature matrix B'. 
                 #
                 # We can then evaluate a point x with two ways:
                 # via ip(g,x)-.5*ip(x,Bx)   {1}
                 #     ip(g,x)-.5*ip(x,B'x)  {2}
                 #
                 # 
                 # CG by definition always decreases {1}. But {2} need
                 # not decrease at each step. The point of
                 # backtracking is that we'll update the parameters
                 # with the x which approximately minimizes {2} to
                 # prevent "overfitting to a minibatch". So if CG
                 # produces M points, x_1,...,x_M, we'd use x_N, where
                 # N<=M. However, we would initialize the next HF step
                 # with x_M, and not x_N.  The idea / hope is to use
                 # an initialization that has more 2nd order
                 # information in it.
                 #
                 # Finally, notice that BT_batches is set to None. If
                 # so, and if grad_batches is large enough, the
                 # default will set BT_batches to [1], to make it
                 # distinct from GN_batches, which is set to [0] by
                 # default.  However, if train_batches==[0], then it
                 # is impossible to set GN_batches to [1], since the
                 # data_object will complain that a minibatch with
                 # index 1 doesn't exist. So if train_batches==[0],
                 # we'll set BT_batches to [0].
                 # 
                 # This is what happens when BT_eval is 'model'.
                 # However, if BT_eval is objective', we evaluate
                 # the partial solution using the objective on the
                 # BT_batches. 
                 BT_batches=None,
                 BT_eval='model', # in 'model', 'objective'


                 # The line search is our mainly safety net. If HF
                 # decided to do something overly ambitious, the line
                 # search will slowly decrease the stepsize of the
                 # proposed update until the objective stops being
                 # unhappy.  It is important, because the quadratic
                 # approximation can be extremely inaccurate in some
                 # of the RNN problems.
                 # 
                 # Our line search likes large step sizes. That is, it
                 # stops decreasing the step size as soon as the value
                 # of the new objective is smaller (small is good)
                 # than the value of the old objective. This way, we
                 # take somewhat larger steps.
                 line_search_batches=None,


                 # RHO_batches specify the data used to evaluate the
                 # reduction ratio as in the original HF paper. The
                 # default is to GN_batches, and there is little
                 # reason to change it.
                 RHO_batches=None,


                 
                 # We may want to optimize until the test losses fall
                 # below some threshold. For example, if we care about
                 # the percent of correctly classified examples, we
                 # could do something like test_losses_threshold_fn =
                 # lambda x: x[1]<0.001 this variable requires a
                 # modest amount of caution when used for the
                 # memorization problem. Our version of the
                 # memorization problem uses an output per timestep,
                 # but most of these outputs are really easy to
                 # get. For example, if T=100 and num_bits=5, then
                 # it's very easy to get 5% accuracy on this
                 # problem. See opt.d.seq.utils.__init__
                 # 
                 # 
                 # It is assumed here that the first element in the
                 # vector of losses x[0] (as returned by losses_fn) is
                 # the thing the optimizer worries about, while the
                 # second element x[1] is the zero one loss that we
                 # care about.
                 # 
                 # And remember: test_losses_threshold_fn is a
                 # function of the losses, returning a bool.
                 test_losses_threshold_fn=None,


                 


                 # the following variables are used for controlling
                 # the runs of CG. mxn/max_cg_iter are obvious. 
                 # cg_eval_freq is the frequency with which we evaluate
                 # BT_batches, and how often we allow ourselves to stop.  
                 min_cg_iter=1,
                 max_cg_iter=100,
                 cg_eval_freq=10,


                 # a parameter for stopping CG based on an "early stopping" criterion.
                 # see the code of get_new_direction for more comments.
                 stop_thresh_factor=1,


                 # we shrink the previous solution by cg_shrink_factor 
                 # before using it as the initialization to the current HF step.
                 cg_shrink_factor=.95,

                 # the maximal number of allowable cg_steps, after
                 # which HF terminates.
                 maxnum_cg = int(1e6),


                 # the maximal number of allowable HF steps after which
                 # we terminate.
                 maxnum_iter = int(5e4),

                 # how often to save the parameter (if not None)
                 # "save every $save_freq HF steps"
                 save_freq=None):



        # most of the parameters below will simply store the above
        # parameters in the instance self.
        self.path = path

        self.message = message
        self.W = init_W
        self.losses_fn = losses_fn
        self.grad_losses_fn = grad_losses_fn
        self.gauss_newton_fn = gauss_newton_fn
        self.data = self.data_object = data_object

        self.init_damping = init_damping

        self.damping = 1*self.init_damping


        if grad_batches is None:
            grad_batches = data_object.train_batches

        self.grad_batches = grad_batches
        self.GN_batches = GN_batches

        # dealing with BT_batches; it's usually [1],
        # unless [1] isn't even a minibatch, in which case it's [0].
        if BT_batches is None:
            if 1 in data_object.train_batches: 
                BT_batches = [1]
            else:
                BT_batches = [0]

        self.BT_batches = BT_batches

        assert BT_eval in ['model', 'objective']
        self.BT_eval = BT_eval


        if line_search_batches is None:
            if 2 in data_object.train_batches:
                line_search_batches = [2]
            else:
                # otherwise our train_batches are so tiny that there's
                # no point in not doing linesearches over the entire dataset. 
                line_search_batches = data_object.train_batches

        self.line_search_batches = line_search_batches


        if RHO_batches is None:          RHO_batches = GN_batches[:]
        self.RHO_batches = RHO_batches
        

        # if we don't specify a test_losses_threshold_fn, we'll
        # pretend it doesn't exist and never stop the optimization:
        if test_losses_threshold_fn is None:
            test_losses_threshold_fn = lambda x: False
        self.test_losses_threshold_fn = test_losses_threshold_fn


        ## set up the printf functions which will write to our logs.
        from opt.utils.printing import make_cg_printf
        self._printf, self._printf_cg = make_cg_printf('runs/' + path.replace('.','/'))
        def printf(x, detail=False):
            if detail is False:
                self._printf(x)
                self._printf_cg(x)
            else: 
                self._printf_cg(x)

        self.printf = printf
        self.printf('\n%s\n\n' % self.message)
        

        # Next initialize the parameters and 
        # CG_x, which is the current solution of CG.
        import gnumpy as g
        self.X = g.garray(init_W.pack())
        self.CG_x = self.X*0

        self.min_cg_iter = min_cg_iter
        self.max_cg_iter = max_cg_iter
        self.stop_thresh_factor = stop_thresh_factor
        self.cg_eval_freq = cg_eval_freq
        


        self.cg_shrink_factor = cg_shrink_factor

        self.maxnum_iter = maxnum_iter
        self.maxnum_cg = maxnum_cg

        self.save_freq = save_freq


        self._total_num_cg = 0
        self._total_batch = 0
        self.iter = 1
        self.test_losses = -1
        
################################################################

    def _op(self, batches, func):
        """
_op takes a list of batch indicies and a function, applies the function
to map(data_object, batches), sums the results, and divides
the result by the total batch size, sum(data_object.size, batches).

We use map(data_object, batches) because batches is, e.g., [1,2,3,4,5],
but map(data_object, batches) is [<data object 1>,...,<data object 5>].
"""
        # We keep track of the total number of batches processed.
        self._total_batch += 1

        # This function increments ans by b, where b is returned
        # from func. This function is somewhat involved because
        # func can either return an array or a list of arrays/scalars.
        def incr(ans,b):
            import gnumpy as g
            if ans==[]:
                if isinstance(b, (np.ndarray, g.garray)):
                    ans.append(1*b)
                else:
                    for bx in b:
                        ans.append(bx)
            else:
                if isinstance(b, (np.ndarray, g.garray)):
                    ans[0]+=b
                else:
                    assert len(ans)==len(b)
                    for (ax,bx) in zip(ans,b):
                        ax+=bx

        # we'd then need to divide this tuple/scalar/array
        # by the total batch size. div will do the job.
        def div(ans, c):
            import gnumpy as g
            for i in range(len(ans)):
                ans[i]/=c
            if len(ans)==1 and isinstance(ans[0], (np.ndarray, g.garray)):
                return ans[0]
            else:
                return tuple(ans)



        ans = []
        tot = 0.


        ## finally,
        for b in batches:
            ## get the data
            d = self.data_object(b)

            ## keep track of the total batch size
            tot += self.data_object.size(d)

            ## and add the function's contribution to the answer.
            incr(ans, func(b,d))

        return div(ans,tot)

    def losses(self, batches, X):
        """
Input: a list of batch ids and the parameters and I'll return the sum
of the losses:        
        """
        def func(b,d):
            return self.losses_fn(d, X)
        return self._op(batches, func)



    def grad(self, batches, X):
        """
Input: a list of batch ids, the parameters
Output: the gradient vector
"""
        def func(b,d):
            return self.grad_losses_fn(d, X)
        return self._op(batches, func)

    def gauss_newton(self, batches, X, R):
        """
Input: a list of batch ids, the parameters, and the vector R
that will be multiplied by the gauss netwon matrix determined by X and batches.
Note that dim(X)==dim(R).
"""


        ## alternate the order in which the minibatches are evaluated.
        ## this makes sense if len(GN_batches)==2. gauss_newton_fn
        ## is supposed to try to cache the state of the net's hidden units.
        ## If we change the data, gauss_newton needs to recompute the hidden units.
        ## If we alternate the order of the minibatches, we'd be giving gauss_newton
        ## 0,1,1,0,0,1,1,0,0,... thus eliminating half of the recomputations
        ## of the hidden state.
        ## gauss_newton is the only function that cares about the order.

        if 'order' not in self.__dict__:
            self.order = True
        self.order = not self.order

        if self.order:
            batches = batches[::-1]

        def func(b,d):
            return self.gauss_newton_fn(b, d, X, R, self.damping)
        return self._op(batches, func)


################################################################
    def update_batches(self):
        """
update_batches is something we call at the end of each HF step.

data_object is told to forget all about its train_dict and to
regenerate its training data anew. This way, successive HF steps work
with different data.

gauss_newton_fn is asked to forget the value of the hidden units it
has cached. 
"""
        self.data_object.forget()
        self.gauss_newton_fn('forget')
################################################################



    def get_new_direction(self, grad):
        """
get_new_direction.

Explicit inputs: grad
Implicit inputs: self.damping[0], self.GN_batches, self.RHO_batches, self.X

We then run CG initialized from the previous solution with no preconditioner.
We stop CG when either:
 - the progress heuristic is activated, or
 - we reach the maximal allowed number of CG steps, or
 - the value of the objective (or loss -- I use the two interchangeably) on 
   GN_batches starts increasing. 


We also evaluate the solution every self.cg_eval_freq 

- EITHER via the quadratic approximation 
$$-grad'delta + 1/2 delta'gauss_newton(x;self.BT_batches) delta$$ (1)

(which differs from the original quadratic approximation only in
self.BT_batches.  After terminating the HF run, we pick the delta that
has the smallest value of (1).  However, we initialize the next step
of HF with the last step of the run of CG. It was helpful for
curves.)

- OR we evaluate it using $loss(x; self.BT_batches)$ if self.BT_eval is 'objective'.

In the above equations, delta is the update computed by CG. However, the code
below calls it x. 
"""
        b = -grad


        # the float typecasting: if damping is a numpy array, then
        # damping[0] is a numpy.float64, which will cause gnumpy to be
        # unhappy.
        self.damper = float(self.damping[0]) #

        # the matrix-vector product:
        def A(x, batches=None):
            if batches is None: 
                batches=self.GN_batches

            return self.gauss_newton(batches, self.X, x) + x*self.damper

        # the preconditioner that does nothing: 
        def M_inv(x):       return x  


        GN_batches = self.GN_batches


        ##  x0 is the initial solution of CG.  self.CG_x is result of
        ## the previous run of CG. We set the latter to the former.
        x0 = self.CG_x


        def model(x, batches=None, r=None, damp=None):
            """
Compute the quadratic model. If the residual r is known, we can
do it quickly, otherwise we'd need to multiply by the curvature 
matrix A.
"""
            assert damp is not None
            assert (r is None)^(batches is None)

            if r is None:
                r = b - A(x,batches)

            ans = -ip(b+r,x)*0.5
            if damp is False: 
                ans -= ip(x,self.damper*x)*0.5
                
            return ans
    
        
        def loss(x, batches):
            """
Compute the loss that we care about. This loss is used for
computing rho and for early stopping.
"""
            return self.losses(batches, self.X+x)[0]



        _reduction_zero_dict = dict()
        def obj_reduction(x, batches):
            """
Compute the reduction in the losses; i.e., compute
loss(x, batches)-loss(x*0, batches). Thus if x is making progress,
obj_reduction should be negative.
"""
            
            # a caching mechanism: to compute the reduction in the loss,
            # we need to know the value of the loss at x=0 on the given batches.
            # so we'll use this _reduction_zero_dict to remember the value
            # of the loss at zero for each value of batches. 
            batches = tuple(batches)
            if batches not in _reduction_zero_dict:
                _reduction_zero_dict[batches] = loss(0, list(batches))
            return loss(x, list(batches)) - _reduction_zero_dict[batches]


        ## track the losses of the quadratic approximation,
        ## so that we'd know when to stop according to the progress
        ## heuristic.
        model_losses = [] 


        ## CG-backtracking related variables
        BT_x = 1*x0
        BT_best = np.inf
        BT_cur = None
        BT_i = None
        BT_model = None

        #### for early-stopping cg: again, this enhancement
        #### isn't actually used in the RNN experiments.
        STOP_best = np.inf
        STOP_cur = None



        ################ start CG
        x = x0 # <-- this is our initialization; notice how it affects r.
        r = b-A(x)
        Mr = M_inv(r)
        d = 1*Mr

        import time
        start_cg = time.time()

        for i in range(self.max_cg_iter):

            print 'running CG: i=%s' % i

            ## begin CG: this is pure textbook CG. Nothing unusual here.
            Ad=A(d) 
            dAd = ip(d,Ad)
            alpha = ip(r,Mr)/dAd

            beta_a=ip(r,Mr)

            ## BEGIN i=i+1
            x += alpha*d
            r -= alpha*Ad
            Mr = M_inv(r)
            ## END 

            beta_b=ip(r,Mr)

            d *= beta_b/beta_a
            d += Mr
            ################ end CG. 

            ## count the number of CG steps
            self._total_num_cg += 1

            ## keep track of the losses
            model_losses.append(model(x,r=r,damp=True))

            ## Stop right away if the residual is tiny.  When a
            ## residual is allowed to become so small, CG can do
            ## something numerically stupid. This happens only at the
            ## beginning of the optimization.
            small_res = norm(r)<1e-10


            ## Stop if the progerss heuristic fires
            no_progress = progress_heuristic(model_losses)

            ## Stop if the objective/loss on GN_batches has gone up.
            STOP_objective_gone_up = False

            ## Normally we stop when check_BT is true, but if the residual
            ## is too small, we'd stop right away.
            gonna_stop_right_now = small_res or no_progress

            ## {variable names act as self-commenting code}
            check_BT = (i % self.cg_eval_freq == 0)


            ## what if we decided to stop right now or to evaluate our 
            ## solution for backtracking: 
            if check_BT or gonna_stop_right_now:

                # get the BT objective: it's the quadratic model on self.BT_batches
                if self.BT_eval is 'model':
                    BT_cur = model(x, batches=self.BT_batches, damp=True)
                else:
                    BT_cur = obj_reduction(x, batches=self.BT_batches)

                # Pay attention if BT_cur is better than BT_best
                if BT_cur < BT_best:
                    BT_best = BT_cur
                    BT_x[:] = x
                    BT_i = i
                    BT_model = model_losses[-1]

                ## keep track of the reduction in the objective function. For kicks.
                cur_loss = obj_reduction(x, self.GN_batches)


                ## keep track of the value of the objecive/loss but
                ## note that we are looking at the reduction rather
                ## than the loss.
                STOP_cur = obj_reduction(x, self.GN_batches)

                ## are we doing well? 
                if STOP_cur < STOP_best:
                    STOP_best = STOP_cur


                # stop_thresh_factor gives us a little bit of "lee-way" 
                # for deciding when to stop CG. That is, if stop_thresh_factor=1,
                # we'll stop the moment the loss goes up. This part could have been
                # easily implemented even if STOP_cur used loss. 
                # However, if stop_thresh_factor=0.5, we'll stop only when the size
                # of the reduction is worse than half of the best reduction. That's also
                # the reason for using obj_reduction for STOP_cur.

                # However, don't use any of the stop_thresh_factor
                # things the reduction is positive (which is a result
                # of a bad initialization, for example).
                if STOP_best < 0:
                    factor = self.stop_thresh_factor
                else:
                    factor = 1

                if (STOP_cur > STOP_best * factor
                    and 
                    # we can also decided that we don't want to turn on this 
                    # early stopping criterion for min_cg_iter CG steps.
                    i >= self.min_cg_iter):
                        STOP_objective_gone_up = True
                        gonna_stop_right_now = True


                
                ## we also use check_BT as an opportunity to print out 
                ## statistics about our CG run. 
                end_cg = time.time()
                tot_cg_time = end_cg - start_cg
                time_per_cg_iter = tot_cg_time / ((i+1))

                # print things:
                titles = ('i', 'model loss', 'loss', 'BT_cur','BT_best',
                          '|x|', '|r|', '|x|_A', 'sec/CG iter')

                values = (i, model_losses[-1], cur_loss, BT_cur, BT_best,
                          norm(x), norm(r),  np.sqrt(-2*ip(r-b,x)), time_per_cg_iter)

                titles_template = '|' + '|'.join([' %12s ']*len(titles)) + '|' + '\n'
                values_template = '|' + '|'.join([' %12.6f ']*len(values)) + '|' + '\n'

                # print the header at the beginning
                if i==0:
                    self.printf('_'*(len(titles_template % titles)-1)+'\n', detail=True)
                    self.printf(titles_template % titles, detail=True)
                self.printf(values_template % values, detail=True)


            # Finally, do the stopping, with an appropriate error message:
            if gonna_stop_right_now:
                self.printf('-'*(len(titles_template % titles)-1)+'\n', detail=True)
            if no_progress:
                self.printf('CG:X: no_progress\n')
            if small_res:
                self.printf('CG:X: small res\n')
            if STOP_objective_gone_up:
                self.printf('CG:X: stopping obj has gone up. Stopping.\n')
            if gonna_stop_right_now:
                break


        ## BT_x kept track of the best solution. We'll use that and
        ## also evaluate rho on that.

        new_x = BT_x
        ## evaluating the reduction ratio: compute the numerator and
        ## the denumenator, then divide.
        RHO_nom = obj_reduction(new_x, self.RHO_batches)
        RHO_denom = model(new_x, batches = self.RHO_batches, damp=True)

        def calc_rho(obj, model):
            import numpy as np
            if obj > 0:
                return -np.inf
            if model == 0:
                return -np.inf
            return obj/model

        RHO = calc_rho(obj =  RHO_nom, 
                       model = RHO_denom)

        ## print the conclusions of our CG run:
        self.printf('CG: rho:%12.6f = nom=%12.6f / denom=%12.6f\n' 
                    % (RHO, RHO_nom, RHO_denom))
        self.printf('CG:               model       denom=%12.6f\n' 
                    % (BT_model))


        self.printf('CG: |bt_x|=%12.6f, |x|=%12.6f\n' % (norm(new_x), norm(x)))
        self.printf('BT/CG = %d/%d\n' % (BT_i, i))

        
        ## we return new_x as our direction, but notice that self.CG_x
        ## is set to the last value of CG.
        return new_x, RHO


    def line_search(self, v):
        """
The line search: choose a maximal 0 <alpha <= 1, so that loss(x+v*alpha)<loss(x)
It's self explanatory.
"""
        def LS_loss(x):
            return self.losses(self.line_search_batches, self.X + x)[0]

        LS_loss_0 = LS_loss(0)
        def LS_red(x):
            return LS_loss(x) - LS_loss_0

        self.printf('line_search: ')
        distances = [.8**i for i in range(50)] + [0]
        assert distances[0]==1 

        for i, step in enumerate(distances): 
            cur_red = LS_red(step*v)
            if cur_red < 0: # a reduction
                self.X += step*v
                break # so stop.

        self.printf('%s linesearches, step = %4.3f, cur_red=%12.6f\n' % (i, step, cur_red))

        ## This is a minor precaution which never happens on curves but can 
        ## happen on some of the RNNs. Basically, if there was no reduction
        ## in the objective even when the stepsize is truly tiny, 
        if step == 0:
            self.printf('line_search: NOTE:setting CG_x to zero because we chose a '
                        'stepsize of 0 for the line_search.\n')
            self.CG_x *= 0


    def optimize(self):
        print 'Starting the optimization...'
        try:
            for self.iter in range(self.iter, self.maxnum_iter):
                self.update_batches()

                if self.save_freq is not None:
                    if self.iter == 1:
                        self.printf('initial saving, to make sure everything works.')
                        self.save()


                self.printf('\n\n\nHF: iter = %s\n' % self.iter)
                print 'computing the gradient and the loss on grad_batches...'
                grad, train_losses = self.grad(self.grad_batches, self.X)

                print 'computing test losses...'
                self.test_losses = self.losses(self.data_object.test_batches, self.X)

                if self.test_losses_threshold_fn(self.test_losses):
                    self.printf("HF: SUCCESS. TEST_LOSSES BELOW THRESHOLD\n")
                    return


                ## useful information.
                self.printf('HF:X: |grad|  =%8.5f\n' % norm(grad))
                self.printf('HF:X: |self.X|=%8.5f\n' % norm(self.X))
                self.printf('HF:X: train    = %s\n'   % train_losses)
                self.printf('HF:X: test     = %s\n'   % self.test_losses)
                self.printf('HF:X: overfit  = %s\n'   % (self.test_losses - train_losses))
                self.printf('HF:X: damping =  %s\n' % self.damping)

                print 'about to start running CG...'
                new_direction, RHO = self.get_new_direction(grad)

                self.damping = damping_heuristic(self.damping, RHO)

                self.line_search(new_direction)
    
                self.CG_x *= self.cg_shrink_factor        

                self.printf('HF: tot_batch = %s; num_cg = %s.\n'
                       % (self._total_batch, self._total_num_cg))

                if self._total_num_cg > self.maxnum_cg:
                    self.printf('HF: self._total_num_cg > self.maxnum_cg. It\'s a good time to stop.\n')
    
                if self.save_freq is not None:
                    if self.iter % self.save_freq == 0:
                        self.save()
                        

        except KeyboardInterrupt:
            self.printf('Ctrl-C: stopping.\n')
            print 'Ctrl-C: stopping.\n'
        except:
            self.printf('an error has occurred.')
            raise


    def save(self):
        "save the parameters to self.path.X"
        from opt.utils import save
        self.printf('saving...\n')
        save(dict(X = self.X.asarray(),
                  iter = self.iter,
                  CG_x = self.CG_x.asarray(),
                  damping = self.damping,
                  _total_num_cg = self._total_num_cg),
             self.path + '.X')
        self.printf('done saving.\n')

    def load(self, path=None):
        "load the parameters from self.path.X or the path parameter"
        from opt.utils import load
        if path is None:
            path = self.path + '.X'
        ans = load(path)
        
        self.X[:] = ans['X']
        self.CG_x[:] = ans['CG_x']
        self.damping[:] = ans['damping']
        self.iter = ans['iter']
        self._total_num_cg = ans['_total_num_cg']
