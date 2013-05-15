"""
A less well-documented implementation of a forward pass neural net 
that's good for HF. It's different somewhat from the RNNs you've
seen so far, in that the batechs that it wants are not (V, O, M), but 
rather (V, O), where V and O are gnumpy 2D arrays, rather than lists of 
2D arrays. It will be better documented in the future. 
"""

import gnumpy as g
import opt.utils.nonlin as u

class FNN:
    def __init__(self, sizes, hid_nonlins, out_nonlin, init=True):
        self.sizes = sizes

        assert len(hid_nonlins) == len(sizes) - 2

        self.out_nonlin = out_nonlin
        self.hid_nonlins = hid_nonlins
        self.nonlins = hid_nonlins + [u.Lin]


        from opt.utils.nonlin import Sigmoid




        if init:
            self.W = [g.randn(v, h) for (v, h) in zip(sizes[:-1],
                                                      sizes[1:])]
            self.b = [g.randn(1, h) for h in sizes[1:]]

            from pylab import sum
            self.size = (sum([w.size for w in self.W]) + 
                         sum([b.size for b in self.W]))

    def initialize_self_weights(self, num_in, scale_big, scale_small=0, W_to_exclude=[]):

        from opt.utils.extra import sparsify_strict

        for i, param in enumerate(self.W):
            if i not in W_to_exclude:
                param[:] = g.randn(*param.shape)
                sparsify_strict(param, num_in, scale_big, scale_small)

        return self

    def initialize_self_biases(self, scale, b_to_exclude=[]):
        for i, b in enumerate(self.b):
            if i not in b_to_exclude:
                b *= scale

    def pack(self):
        return g.concatenate([x.ravel() for x in self.W] + 
                             [y.ravel() for y in self.b])

    def unpack(self, X):
        ans = FNN(self.sizes, self.hid_nonlins, self.out_nonlin, 
                  init = False)
        ans.size = self.size
        from opt.utils.extra import unpack
        mats = unpack(self.W + self.b, X)

        ans.W = mats[:len(self.sizes) - 1]
        ans.b = mats[len(self.sizes) - 1:]

        return ans

    def forward_pass(self, V):
        if type(V) == tuple and len(V)==2:             V = V[0]


        state = [None] * len(self.sizes)

        state[0] = V
        for i in range(len(self.sizes) - 1):
            X = state[i]
            state[i + 1] = self.nonlins[i](g.dot(X, self.W[i]) + self.b[i])

        return state


    def R_forward_pass(self, state, R):


        self.R_state_X = R_state_X = [None] * len(self.sizes)

        R_state_X[0] = state[0]*0

        R_state_i = R_state_X[0]
        for i in range(len(self.sizes) - 1):
            R_state_X[i+1] =  g.dot(state[i], R.W[i]) + \
                              g.dot(R_state_i, self.W[i]) + R.b[i]

            R_state_i = self.nonlins[i].grad_y(state[i+1]) * R_state_X[i+1]

        return R_state_X[-1]




    def backward_pass(self, state, dOX, compute_grad2 = False):
        grad = self.unpack(self.pack() * 0)

        if compute_grad2:
            grad2 = self.unpack(self.pack() * 0)
        else:
            grad2 = None

        dY = dOX
        for i in reversed(range(len(self.sizes) - 1)):
            dX = self.nonlins[i].grad_y(state[i + 1]) * dY

            X = state[i]

            #state[i + 1] = self.hid_nonlin(g.dot(X, self.W[i]) + self.b[i])
            grad.b[i] += dX.sum(0)
            grad.W[i] += g.dot(X.T, dX)

            if compute_grad2:
                grad2.b[i] += (dX*dX).sum(0)
                grad2.W[i] += g.dot((X*X).T, dX*dX)


            ## backprop the gradient:
            if i > 0: # typically the first multiplication is the costliest.
                dY = g.dot(dX, self.W[i].T)

        return grad, grad2




    def preds(self, I):
        state = self.forward_pass(I)
        return state[-1]

    def loss(self, (I, O)):
        OX = self.preds(I)
        loss = self.out_nonlin.loss(OX, O, M=1)
        return loss, OX

    def grad(self, (I, O), compute_grad2=False):
        state = self.forward_pass(I)
        OX = state[-1]

        loss = self.out_nonlin.loss(OX, O, M=1)

        dOX = self.out_nonlin.grad(OX, O, M=1)

        P = self.out_nonlin(OX)

        grad, grad2 = self.backward_pass(state, dOX, compute_grad2) 
        return grad, grad2, loss, OX



    def gauss_newton(self, (I, O), R, state=None, H_damping=None):
        if state is None:
            state = self.forward_pass(I)

        R_OX = self.R_forward_pass(state, R)
        OX = state[-1]

        P = self.out_nonlin(OX)
        try:
            if H_damping is not None:
                LJ = self.out_nonlin.H_prod(R_OX, P, O, M=1, H_damping=H_damping)
            else:
                LJ = self.out_nonlin.H_prod(R_OX, P, O, M=1)
        except TypeError:
            if H_damping is not None:
                LJ = self.out_nonlin.H_prod(R_OX, P, M=1, H_damping=H_damping)
            else:
                LJ = self.out_nonlin.H_prod(R_OX, P, M=1)

        return self.backward_pass(state, LJ)[0]



    
    def __repr__(self):
        return '\n'.join(['FNN3---sizes = %s' % self.sizes.__repr__()])


