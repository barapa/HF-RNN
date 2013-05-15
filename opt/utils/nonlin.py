import gnumpy as g
import numpy as np

class _Softmax_cls:
    def __repr__(self):
        return 'The Softmax Nonlinearity'

    def __call__(self, X):
        "apply the nonlinearity"
        Y = X - X.max(1).reshape(-1,1)
        Z = Y.exp() 
        return Z / Z.sum(1).reshape(-1,1)

    def grad_y(self, Y):
        raise NotImplementedError ("softmax cant yet do gradients wrt Y. Fix it.")

    def grad_Jyx(self, Y, X):
        "multiply by the jacobian of the softmax. This jacobian is symmetric."
        return Y*(X - (X*Y).sum(1)[:,g.newaxis])

    def H_prod(self, X, P, M, H_damping=None):
        "multiply by the little Hessian of the cross entropy, which incidentally is the jacobian above"
        
        assert H_damping is None
        return P * (X - (P*X).sum(1)[:,g.newaxis]) * M

    def logZs(self, XP):
        "get the log partition functions of the softmaxes in the minibatch"
        M = XP.max(1).reshape(-1,1)
        X = XP - M
        return X.exp().sum(1).log().reshape(-1,1) + M

    def loss(self, XP, O, M):
        "get the cross entropy"
        return -((XP - self.logZs(XP)) * O * M).sum()

    def losses(self, XP, O, M):
        return -((XP - self.logZs(XP)) * O * M).sum(1)

    def grad(self, XP, O, M):
        "get the gradient"
        P = self(XP)
        return (P - O) * M

    def grad_grad(self, D_GRAD, XP, O, M):
        P = self(XP)
        return M * P * (D_GRAD - (D_GRAD*P).sum(1)[:,g.newaxis])

    def grad_R(self, RXP, XP, O, M):
        P = self(XP)
        RP = P * (RXP - (P*RXP).sum(1)[:,g.newaxis])
        return RP * M


Softmax = _Softmax_cls()

class _Sigmoid_cls:
    def __repr__(self):
        return 'The Sigmoid Nonlinearity'

    def __call__(self, X):
        return g.sigmoid(X)
    def grad_y(self, Y):
        return Y * (1 - Y)

    def grad_Jyx(self, Y, X):
        return X*Y*(1-Y)


    def grad_y_R(self, H, RH):
        return RH - 2*H*RH

    def grad_y2(self, Y):
        return Y * (1 - Y) * (1 - 2*Y)

    def grad_y2_raw(self, Y):
        return (1-Y) - Y

    def H_prod(self, X, P, M, H_damping=None):
        return X * P * (1-P) * M

    def H12_prod(self, X, P, M, H_damping=None):
        return X * ((P * (1-P))**0.5) * M

    def H_inv_prod(self, X, P, M):
        return X / (P * (1-P)) * M

    def loss(self, XP, O, M):
        return ((O*g.log_1_sum_exp(-XP) + (1-O)*g.log_1_sum_exp(XP))*M).sum()

    def grad(self, XP, O, M):
        P = self(XP)
        return (P - O) * M

    def grad_R(self, RXP, XP, O, M):
        P = self(XP)
        return RXP * M * P * (1-P)

    def grad_grad(self, D_GRAD, XP, O, M):
        P = self(XP)
        return D_GRAD * M * P * (1-P)

    def sample(self, OX):
        return g.rand(*OX.shape) < OX.sigmoid()



Sigmoid = _Sigmoid_cls()

class Squared_Sigmoid:
    def __init__(self, thresh):
        assert thresh > 0
        self.thresh = thresh

    def __repr__(self):
        return ('The Sigmoid Nonlinearity with the squared loss (thresh=%s)' 
                % self.thresh)

    def __call__(self, X):
        return g.sigmoid(X)

    def grad_y(self, Y): #still counts
        return Y * (1 - Y)

    def grad_y2(self, Y):
        raise NotImplementedError

    def H_prod(self, X, P, O, M=1):
        #return X * P * (1-P) * M
        T = O
        H = P * (1-P) * (2*P - 3*P*P - T + 2*T*P)
        th = self.thresh
        H_ = H*(H>th) + th*(H<=th)
        return M * H_ * X

    def H_inv_prod(self, X, P, M):
        raise NotImplementedError

    def loss(self, XP, O, M):
        P = XP.sigmoid()
        D = P-O
        return 0.5*(D*D*M).sum()

    def grad(self, XP, O, M):
        P = XP.sigmoid()
        #return (P - O) * M
        return (P-O)*P*(1-P)*M





class _True_Sigmoid_cls:
    def __call__(self, X):
        return g.sigmoid(X)
    def grad_y(self, Y):
        return Y * (1 - Y)

    def H_prod(self, X, P, M):
        return X * P * (1-P) * M

    def H_inv_prod(self, X, P, M):
        return X / (P * (1-P)) * M

    def loss(self, XP, O, M):
        def H(X):
            from gnumpy import log
            return -(X*log(X+1e-10) + (1-X)*log(1-X+1e-10))
        
        return ((O*g.log_1_sum_exp(-XP) + (1-O)*g.log_1_sum_exp(XP) - H(O))*M).sum()

    def grad(self, XP, O, M):
        P = self(XP)
        return (P - O) * M

True_Sigmoid = _True_Sigmoid_cls()


class _Tanh_cls:
    def __repr__(self):
        return 'The Tanh Nonlinearity'

    def __call__(self, X):
        return g.tanh(X)

    def grad_Jyx(self, Y, X):
        return X*(1-Y*Y)

    def grad_y(self, Y):
        return 1 - Y*Y

    def grad_y_R(self, Y, RY):
        return - 2*Y*RY


    def grad_y2(self, Y):
        return  - 2*Y*self.grad_y(Y)

    def grad_y2_raw(self, Y):
        return -2*Y

    def H_prod(self, X, P, M):
        return X * (1 - P*P) * M * 0.25

    def loss(*args):
        raise NotImplementedError("Tanh_cls: loss is undefined.")

    def grad(*args):
        raise NotImplementedError("Tanh_cls: tot_inp grad is unimplemented.")

Tanh = _Tanh_cls()


class _Lin_cls:
    def __repr__(self):
        return 'The Linear Nonlinearity'

    def grad_y2_raw(self, Y):
        return 0

    def grad_y_R(self, H, RH):
        return 0

    def grad_R(self, RXP, XP, O, M):
        return RXP * M


    def grad_Jyx(self, Y, X):
        return X



    def __call__(self, X):
        return X
    def grad_y(self, Y):
        return 1
    def H_prod(self, X, P, M):
        return X * M
    def H_inv_prod(self, X, P, M):
        return X * M
    def loss(self, XP, O, M):
        D = (O - XP)
        return (D * D * M).sum() * 0.5
    def grad(self, XP, O, M):
        P = XP
        return (P - O) * M
    def grad_y(self, Y):
        return Y*0+1
    def grad_y2(self, Y):
        return Y*0

    def grad_grad(self, D_GRAD, XP, O, M):
        return D_GRAD * M



Lin = _Lin_cls()


# this allows us to stick several different nonlinearities together.
# quite useful in many situations.
class Join:
    def __init__(self, *args):
        assert len(args)%2==0
        fns, ranges = args[::2], args[1::2]
        for (r, r1) in zip(ranges[:-1], ranges[1:]):
            assert r[1]==r1[0]
            assert r[0]<r[1] and r1[0]<r1[1]
        assert r[0]==0

        self.fns = fns
        self.ranges = ranges
        self.size = ranges[-1][1]

    def __repr__(self):
        ans = '\n'.join(['The Join Nonlinearity:'] + 
                        [' nonlin = %s: range = %s' % (fn, rng) 
                         for (fn, rng) in zip(self.fns, self.ranges)])
        return ans
                         


    def __call__(self, X):
        ANS = 0*X
        for (f, (r0,r1)) in zip(self.fns, self.ranges):
            ANS[:, r0:r1] = f(X[:, r0:r1])
        return ANS

    def grad_y(self, Y):
        ANS = 0*Y
        for (f, (r0,r1)) in zip(self.fns, self.ranges):
            ANS[:, r0:r1] = f.grad_y(Y[:, r0:r1])
        return ANS

    def H_prod(self, X, P, M, H_damping=None):
        ANS = 0*X
        for (f, (r0,r1)) in zip(self.fns, self.ranges):
            ANS[:, r0:r1] = f.H_prod(X[:, r0:r1], P[:, r0:r1], M)
        return ANS

    def H_inv_prod(self, X, P, M):
        ANS = 0*X
        for (f, (r0,r1)) in zip(self.fns, self.ranges):
            ANS[:, r0:r1] = f.H_inv_prod(X[:, r0:r1], P[:, r0:r1], M)
        return ANS

    def loss(self, XP, O, M):
        ans = 0
        for (f, (r0,r1)) in zip(self.fns, self.ranges):
            cur_loss = f.loss(XP[:, r0:r1], O[:, r0:r1], M)
            ans  += cur_loss
        return ans

    def grad(self, XP, O, M):
        ANS = 0*XP
        for (f, (r0,r1)) in zip(self.fns, self.ranges):
            ANS[:, r0:r1] = f.grad(XP[:, r0:r1], O[:, r0:r1], M)
        return ANS




class Join_IO:
    """
Join differs from Join_IO in that it allows for the possibility 
of the nonlinearities having different numbers of inputs and outputs.
"""
        

    def __init__(self, *args):
        assert len(args)%2==0
        fns, sizes, weights = args[::3], args[1::3], args[2::3]

        assert len(fns)==len(sizes)==len(weights)

        from pylab import array
        input_sizes = array([x for (x,y) in sizes])
        output_sizes = array([y for (x,y) in sizes])

        self.fns = fns
        self.input_sizes = input_sizes
        self.output_sizes = output_sizes
        self.weights = array(weights)

        self.input_ranges = []
        ptr = 0
        for s in input_sizes:
            a = ptr
            b = ptr + s
            self.input_ranges.append((a,b))
            ptr = b
        assert ptr == input_sizes.sum()


        self.output_ranges = []
        ptr = 0
        for s in output_sizes:
            a = ptr
            b = ptr + s
            self.output_ranges.append((a,b))
            ptr = b
        assert ptr == output_sizes.sum()

        self.input_size = input_sizes.sum()
        self.output_size = output_sizes.sum()

        self.loss_acc = g.zeros(len(fns))



    def __repr__(self):
        ans = '\n'.join(['The Join_IO Nonlinearity:'] + 
                        [' nonlin = %s: input = %s, output=%s' % (fn, input, output) 
                         for (fn, input, output) in zip(self.fns, self.input_sizes, self.output_sizes)])
        return ans
                         
    def __call__(self, X):
        assert X.shape[1] == self.input_size
        ANS = 0*X
        for (f, (ir0,ir1), (or0,or1)) in zip(self.fns, self.input_ranges, self.output_ranges):
            ANS[:, ir0:ir1] = f(X[:, ir0:ir1])
        return ANS

    def sample(self, X):
        assert X.shape[1] == self.input_size
        batch_size = X.shape[0]
        ANS = g.zeros((batch_size ,self.output_size))
        for (f, (ir0,ir1), (or0,or1)) in zip(self.fns, self.input_ranges, self.output_ranges):
            ANS[:,or0:or1] =  f.sample(X[:, ir0:ir1])
        return ANS


    def grad_y(self, Y):
        assert Y.shape[1] == self.input_size
        ANS = 0*g.zeros((len(X),self.input_size))
        for (f, (ir0,ir1), (or0,or1)) in zip(self.fns, self.input_ranges, self.output_ranges):
            ANS[:, ir0:ir1] = f.grad_y(Y[:, ir0:ir1])
        return ANS

    def H_prod(self, X, P, M, H_damping=None):
        assert X.shape[1] == self.input_size
        assert P.shape[1] == self.input_size

        if H_damping is None: H_damping = [None]*len(self.fns)
        ANS = 0*X
        for (f, 
             (ir0,ir1), 
             (or0,or1), 
             h, 
             w) in zip(self.fns, 
                       self.input_ranges, 
                       self.output_ranges, 
                       H_damping, 
                       self.weights):

            ANS[:, ir0:ir1] = f.H_prod(X[:, ir0:ir1], P[:, ir0:ir1], M, H_damping=h)*w

        return ANS

    def H_inv_prod(self, X, P, M):
        raise NotImplementedError

    def loss(self, XP, O, M):
        assert XP.shape[1] == self.input_size
        assert O.shape[1] == self.output_size

        ans = 0
        for (i, (f, 
                 (ir0,ir1), 
                 (or0,or1), 
                 w)) in enumerate(zip(self.fns, 
                                      self.input_ranges, 
                                      self.output_ranges, 
                                      self.weights)):

            # import pdb; pdb.set_trace()

            cur_loss = f.loss(XP[:, ir0:ir1], O[:, or0:or1], M) * w
            ans  += cur_loss
            self.loss_acc[i] += cur_loss

        

        return ans

    def grad(self, XP, O, M):
        assert XP.shape[1] == self.input_size
        assert O.shape[1] == self.output_size

        ANS = XP*0
        for (f, 
             (ir0,ir1), 
             (or0,or1), 
             w) in zip(self.fns, 
                       self.input_ranges, 
                       self.output_ranges, 
                       self.weights):

            ANS[:, ir0:ir1] = f.grad(XP[:, ir0:ir1], O[:, or0:or1], M) * w
        return ANS








class DiagGaussian_generic:
    def __init__(self, f, f_prime_y, log_a_=None, min_H_damping=0.):
        self.min_H_damping = min_H_damping
        self.f = f
        self.f_prime_y = f_prime_y
        if log_a_ is None:
            def log_a_(a_): 
                return f(a_).log()
        self.log_a_ = log_a_

    def __repr__(self):
        return '\n'.join(['Diagonal Gaussian nonlin: uses more outputs than there are inputs.',
                          'f = %s' % self.f,
                          'f_prime_y = %s' % self.f_prime_y,
                          'min_H_damping = %s' % self.min_H_damping])

    def __call__(self, X):
        b,y=X.shape
        assert y%2==0

        
        b = X[:,:y/2]
        a_ = X[:,y/2:]
        a = self.f(a_)

        ans = g.zeros(X.shape)
        ans[:, :y/2] = b/a
        ans[:, y/2:] = a

        return ans

    def grad_y(self, Y):
        raise NotImplementedError()
    
    def H_prod(self, X, P, M=1, H_damping=1):
        b,y = P.shape
        assert y%2==0

        m = P[:, :y/2] # m is first.
        a = P[:, y/2:]
        a_d = self.f_prime_y(a)

        s = 1/a

        R_b = X[:, :y/2]
        R_a_ = X[:, y/2:]


        A_00 =        (s                )
        A_01 = A_10 = (-s*m             )*a_d
        A_11 =        (s*m**2  + .5*s**2)*a_d**2

        ANS = g.zeros(P.shape)
        ANS[:, :y/2] = R_b*A_00 + R_a_*A_01
        ANS[:, y/2:] = R_b*A_10 + R_a_*(A_11 + max(H_damping, self.min_H_damping))

        return ANS*M


    def loss(self, XP, O, M):
        b,y = XP.shape
        assert y%2 == 0
        b = XP[:, :y/2]
        a_ = XP[:, y/2:]
        a = self.f(a_)

        const = 0 #float(np.log(2*np.pi))
        m = b/a
        dd = (O - m)
        #return ((a*dd*dd*0.5 - a.log()*0.5 + const*0.5) * M).sum()
        return ((.5*a*dd*dd - .5*self.log_a_(a_) + const*0.5) * M).sum()


    def grad(self, XP, O, M):
        b,y = XP.shape
        assert y%2 == 0

        b = XP[:, :y/2]
        a_ = XP[:, y/2:]
        a = self.f(a_)
        a_d = self.f_prime_y(a)


        G = g.zeros(XP.shape)

        m = b/a # that's good
        G[:, :y/2] = -(O - m)

        ## try the A gradients: right. 
        s = 1./a

        E_OO = s + m*m
        ## that's good oto.
        G[:, y/2:] = (O*O - E_OO)*0.5 * a_d

        return G * M

    def sample(self, XP):
        y = XP.shape[1]
        assert y%2==0
        b = XP[:, :y/2]
        a_ = XP[:, y/2:]
        a = self.f(a_)

        const = 0 #float(np.log(2*np.pi))
        m = b/a

        # a is the actual precision. or 1/sigma^2. 
        # so 1/a is sigma**2. so 1/a**0.5 is sigma.
        # and we multiply by the standard deviation. 
        std = 1/a**0.5
        return m + randn(*b.shape)*std


def _exp(x): return x.exp()
def _exp_prime_y(x): return x
def _log_a_(x): return x
DiagGaussian_std = DiagGaussian_generic(_exp, _exp_prime_y, _log_a_)
def DiagGaussian_std_min_H_damping(min_H_damping): 
    return DiagGaussian_generic(_exp, _exp_prime_y, _log_a_, min_H_damping = min_H_damping)

def _rect(x): return g.log_1_sum_exp(x)
def _rect_prime_y(y): 
    return ((1-(-y).exp()) * (y>0.1) + 
            y * (y<0.1))

DiagGaussian_rect = DiagGaussian_generic(_rect, _rect_prime_y)
def DiagGaussian_rect_min_H_damping(min_H_damping): 
    return DiagGaussian_generic(_rect, _rect_prime_y, min_H_damping = min_H_damping)



class MogDiag(object):
    def __init__(self, num_components, f, f_prime_y, log_a_):
        self.f = f
        self.f_prime_y = f_prime_y
        self.log_a_ = log_a_
        self.num_components = num_components
    
    def __repr__(self):
        return '\n'.join(['Diagonal Gaussian Mixture of %s components' % self.num_components,
                         'f = %s' % self.f,
                         'f_prime_y = %s' % self.f_prime_y])

    def get_mean(self, X):
        assert (X.shape[1] - self.num_components) % self.num_components == 0
        dim = (X.shape[1] - self.num_components) / self.num_components 

        mixture_weights = X[:, :self.num_components]


        As = []
        Bs = []
        s = self.num_components
        for i in range(self.num_components):
            Bs.append(X[:, s:s+dim]) ; s+=dim # Bs come first.
            As.append(X[:, s:s+dim]) ; s+=dim

        
        mean = 0
        for j in range(self.num_components):
            mean += mixture_weights[:,[j]] * Bs[j]/self.f(As[j]) # the shapes are correct.
               
        return mean

    def __call__(self, X):
        assert (X.shape[1] - self.num_components) % self.num_components == 0
        ## there's no point for the nonlinearity at all. That's cool.
        return X
                         
    def loss(self, XP, O, M):
        ## how is it going to work? we are going to get A and B, 
        ## and then do a partition function trick. We could also put the mixture coefficients
        ## right into this thing. 
        pass
