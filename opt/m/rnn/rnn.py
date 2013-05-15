import gnumpy as g
#import p10.mrnn.utils as u
import opt.utils.nonlin as nonlin
from opt.utils.extra import unpack, sparsify_strict

"""
This file defines the RNN class. 
To be trainable by HF, it needs to support three functions:
the supervised loss, grad, and gauss_newton.

While we are familiar with the supervised loss and the gradient,
the gauss_newton is a different. I will now explain how it works.

The RNN (as well as any other neural net) consists of two parts:
the part that maps the parameters to the predictios (which is the actual
neural net), and the part that takes the predictions and gives us a 
single numerical loss.

Call the first part N:theta -> predictions.  Then J = N' is the
Jacobian of N. It maps changes in parameters delta theta to linearized
changes in the predictions: delta predictions_lin = J delta theta

Now the R-prop implements efficient multiplication by J. That is, if
theta is fixed, and delta theta is given, then R-prop(delta theta)
will precisely compute the linearized change in the prediction. In
effect, R-prop computes a linear forward pass that tries to
approximate the original nonlinear function.

Conveniently, the backward pass multiplies by J' (J transpose). That
is, given dE/dPredictions, we can run them through backprop and get
the change in the parameters.  Well, it so happens that
delta W = W' dE/dPredictions. 

But: to multiply by J'J, we need to take delta that, run it through
R-prop, and send the result to backprop. And that's precisely the
operation of the gauss_newton function below. In reality, we need to
multiply by J'L''J where L'' is the hessian of the function that takes
the predictions and gives us the loss. But usually L'' is a very
simple matrix (diagonal for the sigmoids, diagonal + rank1 for the
softmax), so it's also easy to multiply by it.

In conclusion: run R-prop to get the product by J. Then use a simple
formula to multiply this result by L''. Then send the result to
backprop to obtain the product with J'. That's the algorithm for
multiplying gauss-newton matrix.

The R-prop is backprop's dual in a very strong sense. Backprop uses
the chain rule with respect to the inputs, while R-prop dosen't use
the chain rule per se; R-prop simply computes d variable / d (delta
theta) for each variable as the net does its forward pass and computes
each variable. See the code. You'll see a one-to-one correspondence
between the forward_pass function and the R-forward pass function. 

Future versions will have a clean testing routine that ensure that the
loss, the gradient, and the gauss_newton matrix are consistent. Of
course, the gradient and gauss_newton are completely determined by the
loss function and the structure of the neural net, and can be easily
computed automatically. flex.py and theano can the differentiation 
automatically. 



"""


# Techincal note: grad2, or sum_{data_i} grad(data_i)^2 cannot be 
# easily computed for RNN due to the weight sharing. 
# For deep nets, you'd do grad2.W += outp(Y_i*Y_i, dX_{i+1}*dX_{i+1})
# and that would be correct.
# However, for RNNs, using the above formula would give the following:
# But first, let's set the notation:
# grad = sum_i sum_t grad_{it}, which are the terms that get added as 
# BPTT does its thing.
# where i is the sequence id and t is the timestep.
# The above would give us:
# sum_i sum_t grad_{it}^2. 
# However, we want
# sum_i (sum_t grad_{it})^2.
# Which is something that simply can't easily be computed in the 
# current setup. 

class RNN: 
    def __init__(self, 

                 v, h, o,
                 
                 hid_nonlin=None,

                 out_nonlin=None,
                 
                 struct_damp_nonlin=None,

                 init=True):

        self.v = v
        self.h = h
        self.o = o

        if hid_nonlin is None:
            hid_nonlin = nonlin.Tanh
        if out_nonlin is None:
            out_nonlin = nonlin.Softmax
        if struct_damp_nonlin is None:
            raise TypeError('must specify struct_damp_nonlin.')

        self.hid_nonlin = hid_nonlin
        self.out_nonlin = out_nonlin
        self.struct_damp_nonlin = struct_damp_nonlin

        if init:
            self.h_init = g.randn(1, h)

            self.W_hh = g.randn(h, h)
            self.W_vh = g.randn(v, h)
            self.W_ho = g.randn(h, o)


        if hid_nonlin is None:
            hid_nonlin = nonlin.Tanh
        self.hid_nonlin = hid_nonlin

        if out_nonlin is None:
            out_nonlin = nonlin.Lin

        self.out_nonlin = out_nonlin


    def __repr__(self):
        return '\n'.join(['fast RNN:',
                          ' v=%s' % self.v,
                          ' h=%s' % self.h,
                          ' o=%s' % self.o,
                          ' hid_nonlin = %s' % self.hid_nonlin,
                          ' struct_damp_nonlin = %s' % self.struct_damp_nonlin,
                          ' out_nonlin = %s' % self.out_nonlin])


    def initialize_self(self, num_in, scale, 
                        scale_small=0., 
                        vars_to_exclude=[], vars=None):

        """
The random sparse initialization of the RNN.
        """
        for (var_name, var) in self.__dict__.iteritems():

            if (var_name not in vars_to_exclude and 
                isinstance(var, g.garray)     and
                ((vars is None) or (var_name in vars))):
                    sparsify_strict(var, num_in, scale, scale_small)
                    print('RNN:sparsifying %s\n' % var_name)

        return self



    def pack(self):
        return g.concatenate([self.h_init.ravel(),

                              self.W_hh.ravel(),
                              self.W_vh.ravel(),

                              self.W_ho.ravel()])

    def unpack(self, X):
        ans = RNN(self.v, self.h, self.o,
                  self.hid_nonlin, self.out_nonlin,
                  self.struct_damp_nonlin)

        (ans.h_init,
         ans.W_hh,
         ans.W_vh,
         ans.W_ho) = unpack([self.h_init,
                               self.W_hh,
                               self.W_vh,
                               self.W_ho],
                              X)
        return ans


    
    def forward_pass(self, batch):
        """
Given a batch (V, O, M)  (for M, see opt.d.seq.utils.__init__),
compute the hidden state sequence and the predictions.
"""

        if type(batch) is tuple:
            if len(batch)== 3:
                V, O, M = batch
                assert len(V) == len(O) == len(M)
            elif len(batch) == 2:
                V, O = batch
                assert len(V) == len(O)
            else:
                raise TypeError
        elif type(batch) is list:
            V = batch # strictly speaking, forward_pass only needs V.
        else:         # but I allow it to accept batches, for convenience.
            raise TypeError

        if V[0] is not None:
            V = [None] + V
        


        assert V[0] is None
        T = len(V)-1
        batch_size = len(V[1])

        H, OX = [[None]*(T+1) for _ in range(2)]

        H[0] = g.tile(self.h_init, (batch_size, 1))
        for t in range(1, T+1):
            HX_t = g.dot(H[t-1], self.W_hh) + g.dot(V[t], self.W_vh)
            H[t] = self.hid_nonlin(HX_t)
            OX[t] = g.dot(H[t], self.W_ho)

        return (V[1:], H, OX[1:])



    def R_forward_pass(self, state, R):
        """
Apply the R-operator on RNN. R is an RNN object which represents the
vector we multiply by. Note that it needs to know the RNN's state, so
that it doesn't have to unnecessarily recompute the state.
"""

        V, H, OX = state

        if V[0] is not None:
            V = [None] + V

        assert V[0] is None

        T = len(V)-1
        batch_size = len(V[1])
        R_OX, R_HX = [[None]*(T+1) for _ in range(2)]

        import numpy as np

        R_H_t = g.tile(R.h_init, (batch_size, 1))
        for t in range(1, T+1):
            R_H_1t = R_H_t

            R_HX[t] = g.dot(R_H_1t, self.W_hh) + g.dot(H[t-1], R.W_hh) + g.dot(V[t], R.W_vh)
            R_H_t = self.hid_nonlin.grad_y(H[t]) * R_HX[t]
            R_OX[t] = g.dot(H[t], R.W_ho) + g.dot(R_H_t, self.W_ho)


        #             \/---(for the structured reg).
        return (R_HX, R_OX[1:])


    
    def backward_pass(self, state, dOX, R_HX=None, mu_times_lambda=0.):
        """
The backward pass (or the L-op). Given the gradients wrt the output
units and the state, compute the implied derivative wrt the parameters.
If R_HX is given, then structural damping will be added. 
"""

        V, H, OX = state
        if V[0] is not None:
            V = [None] + V
        if OX[0] is not None:
            OX = [None] + OX
        if dOX[0] is not None:
            dOX = [None] + dOX

            
        assert V[0] is None
        T = len(V)-1

        grad = self.unpack(self.pack() * 0)


        dH_1t = H[-1] * 0 
        for t in reversed(range(1, T+1)):

            dH_t = dH_1t 

            dH_t += g.dot(dOX[t], self.W_ho.T) 
            grad.W_ho += g.dot(H[t].T, dOX[t])


            
            ## backpropagate the nonlinearity: at this point, dHX_t, the gradinet
            ## wrt the total inputs to H_t, is correct.
            dHX_t = dH_t * self.hid_nonlin.grad_y(H[t])

            ## THIS IS THE ONLY LINE THAT HAS ANYTHING TO DO WITH STRUCTURAL
            ## DAMPING. Pretty cool :-)
            if R_HX is not None:
                dHX_t += float(mu_times_lambda) * \
                    self.struct_damp_nonlin.H_prod(R_HX[t], H[t], 1)


            dH_1t = g.dot(dHX_t, self.W_hh.T)

            grad.W_hh += g.dot(H[t-1].T, dHX_t)
            grad.W_vh += g.dot(V[t].T, dHX_t)



        grad.h_init += dH_1t.sum(0)

        return grad



    ### Next, take the above functions and use them to construct the functions
    ### we actually care about. Loss, grad, and gauss_newton.
    def loss(self, (V, O, M)):
        assert len(V) == len(O) == len(M)
        (V, H, OX) = self.forward_pass(V)

        loss = 0
        for t in range(len(O)):
            loss += self.out_nonlin.loss(OX[t], O[t], M[t])

        return loss, OX


    def preds(self, V):
        (V, H, OX) = self.forward_pass([None] + V)
        assert len(OX)==len(H)-1
        return OX

                             
    def grad(self, (V, O, M), loss=False):
        

        state = self.forward_pass(V)

        OX = state[-1]

        dOX = [None] * len(OX)

        loss = 0
        for t in range(len(O)):
            loss += self.out_nonlin.loss(OX[t], O[t], M[t])
            dOX[t] = self.out_nonlin.grad(OX[t], O[t], M[t])

        self.cur_loss = loss

        grad = self.backward_pass(state, dOX)

        return grad, loss, OX




    def gauss_newton(self, data, R, state=None, mu=None):
        """
gauss_newton: if you know the state, we won't need to recompute it and
thus save time. R is the RNN.unpack(vector) we're multiplying by. mu
is the scale of the structural damping, which corresponds to mu*lambda
in the paper.
"""
        (V, O, M) = data
        
        if state is None:
            state = self.forward_pass(V)

        R_HX, R_OX = self.R_forward_pass(state, R)
        
        (V, H, OX) = state

        # multiply by the "little hessian":
        # (H_prod does that).
        LJ = [None] * len(R_OX)
        for t in range(len(OX)):
            P_t = self.out_nonlin(OX[t])
            LJ[t] = self.out_nonlin.H_prod(R_OX[t], P_t, M[t])

        if mu is not None:
            return self.backward_pass(state, LJ, R_HX, mu)
        else:
            return self.backward_pass(state, LJ)
