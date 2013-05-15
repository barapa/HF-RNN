import gnumpy as g
from opt.utils.nonlin import Softmax, Lin
from numpy.random import randint

"""
There is one other thing I didn't mention yet. It's called 
num_outputs. It determines 

"""

def expand(labels, n_labels):
    from numpy import zeros
    a=zeros((len(labels), n_labels))
    for i in range(len(labels)):
        a[i,labels[i]]=1
    return a


# Note how the loss functions ignore the output units
# whose mask is set to zero. 
def zero_one_loss_softmax((X, Y, M), PX):
    """
Return the zero-one loss. Here X is the input 
and is unused. Y are the targets, M are the masks, and PX
are the predictions. We give the function (X, Y, M) as a tuple
to make it natural to give it a batch object, which is precisely
this kind of a tuple. See opt.r.demo

"""
    loss = 0.
    for (px, m, y) in zip(PX, M, Y):
        # m.ravel.asarray: turns out gnumpy.argmax returns a numpy array.
        # So we move m back to the CPU also.
        loss += ((px.argmax(1) != y.argmax(1)) * m.ravel().asarray()).sum()
    return loss



def zero_one_loss_softmax_last((X, Y, M), PX):
    """
Return the zero-one softmax loss only at the last timestep. Here X is the input 
and is unused. Y are the targets, M are the masks, and PX
are the predictions. 

"""
    loss = 0.
    
    px = PX[-1]
    m = M[-1]
    Y = Y[-1]
    loss += ((px.argmax(1) != y.argmax(1)) * m.ravel().asarray()).sum()

    return loss


def zero_one_loss_real_valued((X, Y, M), PX):

        loss = 0.
        for (px, m, y) in zip(PX, M, Y):
            loss += ((abs(px - y) > 0.04) * m).sum()
        return loss

def zero_one_loss_sigm((X, Y, M), PX):
        loss = 0.
        for (px, m, y) in zip(PX, M, Y):
            loss += (((px > 0) != (y > 0.5)) * m).sum()
	    # y \in {0,1}. So px>0 -> y=1
	    # similarly,      px<0 -> y=0. 
	    # The "> 0.5" is to allow for y=.99 or y=.01.
        return loss


class DataAccess(object):
    def __call__(self, x):
        if x > 0:
            mem = self.train_mem
        else:
            mem = self.test_mem
        
        if x not in mem:
            mem[x] = self.make_V_O_M()

        V_np, O_np, M_np = mem[x]

        return (map(g.garray, V_np),
                map(g.garray, O_np),
                map(g.garray, M_np))



    def forget(self):
        self.train_mem.clear()

    def size(self, batch):
        """
How many output units are in the minibatch? 
"""
        x,y,m = batch 
        return sum([mm.sum() for mm in m])


    
    def visualize_batch(self, batch, batch_id, whitespace_color=0.5):
        """
Use that to see what's going on in our data. Did we instantiate
the data object correctly? By looking at the different sequences
in the batch, we get to really see what's going on. 

Suppose we ran an RNN and got a sequence of states, H. We can then 
simply append H to the batch tuple: normally, x,y,m = batch,
but if we instead provide (x,y,m,h), then we'll also get to see the
hidden state sequence. Aren't RNNs cool?!

        """

        def show(x): 
            from pylab import imshow, gray
            gray() # make sure the colors are sensible
            return imshow(x, interpolation='nearest')


        ans = []
        T = len(batch[0])
        for b in batch:
            assert len(b)==T
            from numpy import array, zeros
            cur_image = array([bb[batch_id].asarray() for bb in b]).T
            ans.append(cur_image)

            whitespace = zeros((1, T))+whitespace_color

            ans.append(whitespace)

        from numpy import concatenate
        show(concatenate(ans))



class op_cls(DataAccess):
    """
The op_cls is a general data generator that covers the addition
problem, the multiplication problem, and the xor problem. 
"""
    def __init__(self, 

                 T_min, T_max,  # <-- these problems have variable length sequences

                 A_min, A_max,  # <-- the range of the first operand
                 B_min, B_max,  # <-- the range of the second operand

                 batch_size, 

                 op, # <-- op: computes the target from the two operands.
                     # could be lambda x,y: x+y, lambad x,y:x*y, or anything else.

                 inp_dist, # <-- the input distribution: could be rand, 
                           # randn, or even lambda *shape: rand(*shape)<0.5

                 num_train_batches, 
                 num_test_batches,   
                 # we are free to decide on the number of train and test
                 # batches. 
                 # The number of test batches controls the amount of data
                 # used for evaluation.
                 #
                 # num_train_batches essentially upper bounds the size
                 # on the gradient, because grad_batches \subseteq num_train_batches.
                 # 
                 



                 out_nonlin = None,
                 # the output nonlinearity

                 zero_one_loss = None
                 # the zero-one loss, that determines the success of an experiment
                 ):


        
	assert zero_one_loss is not None
        self.zero_one_loss = zero_one_loss

	assert out_nonlin is not None
        self.out_nonlin = out_nonlin


        self.T_min = T_min;        self.T_max = T_max;        
        self.A_min = A_min;        self.A_max = A_max;
        self.B_min = B_min;        self.B_max = B_max;


        self.batch_size = batch_size
        
        self.V = 2 # In these problems, we always have two inputs 
        self.O = 1 # and one output.


        self.op = op
        self.inp_dist = inp_dist

        self.train_mem = dict() 
        self.test_mem = dict() 
        # an op_cls instance behaves like a function. It gets a
        # batch_id and outputs a batch. When we wish to look at a test
        # batc for the first time, we'll generate this batch and store
        # it in self.test_mem. This way, we randomly choose a fixed
        # test set and work with it. 

        # train_mem is similar but with one difference: 
        # since we wish to evaluate the gradient
        # on different data at each HF iteration, we will clear train_mem
        # at the end of each HF step.

        self.batch_fn = self
        

        self.train_batches = [i for i in range(num_train_batches)]
        self.test_batches = [-i-1 for i in range(num_test_batches)]


        self.T = T_max



        # how to scale various quantities.
        self.num_outputs = 1
        ### self.num_actual_inputs = 2



    def make_V_O_M(self):
        """create a fresh minibatch.
Note that a minibatch consists of three lists of length T.
V, the input list.
O, the target list.
and M, the mask list. The mask list is used to implement variable-length
sequences within a minibatch framework. If we use minibatches, we cannot 
force the sequences to be of physically different lengths. Instead, 
each element of the minibatch (b) at each timestep (t) has a mask, M[t][b]. 
If the mask M[t][b] is 1, then the loss correspoding to this timestep is taken
into account in the evaluation of the total loss and the total gradient. But if 
M[t][b] is 0, then this timestep doesn't contribute to the loss nor to the gradinet. 
I.e., it doesn't exist. So if we want to make the bth sequence be of length 10,
we set M[:10][b] to zero.

NB V, O, and M are lists of arrays of size [batch_size X relevant_size]. V, O, and M
are not simply arrays due to an old bug in cudamat (which might now be fixed) that
prevented gnumpy from successfully working with such tensors when they were large. 
"""



        from numpy import zeros
        V = zeros((self.T_max, self.batch_size, 2))
        O = zeros((self.T_max, self.batch_size, 1))
        M = zeros((self.T_max, self.batch_size, 1))

        for b in range(self.batch_size):
            A, B = self.inp_dist(2) 
            target = self.op(A, B)

            # choose the length of the sequence

            T = randint(self.T_min, self.T_max)



            # first generate the random inputs:
            V[:, b, 0] = self.inp_dist(self.T_max) 
            V[:, b, 1] = 0


            # the very first timestep is equal to -1. See
            # The LSTM paper. This choice wouldn't make any difference
            # on our results. Hopefully.
            V[0, b, 1] = -1

            # the arguments:
            I = randint(self.A_min, self.A_max)
            V[I, b, 0] = A # the value
            V[I, b, 1] = 1 # the mark
            
            J = randint(self.B_min, self.B_max)
            V[J, b, 0] = B # the value
            V[J, b, 1] = 1 # the mark

            # prepare our network to make a prediction:
            # give the net a signal that it's about to make a prediction.
            # I noticed that I've been doing it in the experiments. Probably
            # to be more similar to the original experiments. It
            # probably shouldn't hurt much, but it could do a bit of damage. 
            V[T - 3, b, 1] = -1

        
            # Specify the target at the last timestep
	    # of the sequence, and set the mask to 1 at this
	    # instant.
            M[T, b, 0] = 1
            O[T, b, 0] = target
            # note that there is only one output unit at the end
            # of the sequence.

        return V, O, M













class temporal2(DataAccess):

    """
temproal2 is a general data generator that implements
the temporal order problem with two bits. In a sequence
of random symbols, two symbols are deemed special, because
they are unique, occuring at most twice in the sequence. 
The goal is to output the order and the identities of the special
sequence in the end of the sequence.
"""

    def __init__(
        self, 
        T_min, T_max, # bounds on the length of the sequence
        A_min, A_max, # the range of the first symbol
        B_min, B_max, # the range of the second symbol

        batch_size, 

        num_train_batches=10, # as before; see op_cls
        num_test_batches=10):

	    
	    
        self.zero_one_loss = zero_one_loss_softmax
        self.out_nonlin = Softmax
	# we know the kind of output nonlinearity we should
	# use in the temporal problems: Softmax, and its 
	# corresponding zero one loss.


        self.T_min = T_min;        self.T_max = T_max;        
        self.A_min = A_min;        self.A_max = A_max;
        self.B_min = B_min;        self.B_max = B_max;



        self.batch_size = batch_size
        
        self.V = 8 # note that we have 8 possible inputs becuase there
	           # are 8 possible symbols. We get a dimension per
	           # symbol and use a 1-of-N encoding, as we will see
	           # below.

        self.O = 4 # there are only 4 outputs representing (X,X),
		   # (X,Y), (Y,X), and (Y,Y).

	# Some may wonder, how comes we can solve the temproal order problem
	# but not the Xor? 
	# The reason lies in the incremental nature of this problem,
	# where a sequence looks like this: ...ABAB,X_1,BBBAB....BA,X_2,BA...END
	# Where X_1 and X_2 \in {X,Y}.

	# Let's also assume that it is difficult to notice long range
	# dependencies. In temporal order, we can first notice the
	# relationship between Y and the target, without making any
	# conclusions about X. That is, it is easier to notice that
	# X_2->(?,X_2). So given X_2, we could learn to output (X,X_2)
	# and (Y,X_2) and improve our performance in the process. Once
	# we learn how to send X_2 to the output, our weights become
	# good at transmitting long range information while ignoring
	# noise, which will make it easier to latch on to the first
	# argument.

	# The Xor problem does not have such a luxury, because the
	# second argument alone tells us nothing at all about the
	# first argument.  That's why solving the problem "one
	# argument at a time" does not apply.
	


        self.T = T_max



        self.train_mem = dict()
        self.test_mem = dict()

        # we implement a batch fn in a cheap and a dirty way. Awesome. 
        self.batch_fn = self
        

        self.train_batches = [i for i in range(num_train_batches)]
        self.test_batches = [-i-1 for i in range(num_test_batches)]


	
        self.num_outputs = 8
        ###  self.num_actual_inputs = 1


    def make_V_O_M(self):
        from numpy import zeros
        V = zeros((self.T_max, self.batch_size, self.V))
        O = zeros((self.T_max, self.batch_size, self.O))
        M = zeros((self.T_max, self.batch_size, 1))



	for b in range(self.batch_size):
		
            from pylab import randint
            A = 4+b%2           #randint(4,6)
            B = 4+(b/2)%2       #randint(4,6)

	    # I previously had randints, but randints 
	    # will have unbalanced minibatches: perhaps
	    # 51% of them will have A=X, and 49% will have A=Y.
	    # The above formulas deterministically select
	    # X and Y so to be precisely equally probable. 

            target = 2*(A-4) + (B-4)

            from numpy.random import randint
            T = randint(self.T_min, self.T_max)

            # 1: generate the random symbols; notice that the first
	    # four symbols are random, but we'd like to expand
	    # them to 8-wide arrays. 
            V[:, b, :] = expand(randint(0, 4, size=self.T_max), 8)

	    # the 6th symbol notifies the RNN of the upcoming
	    # prediction.  The RNN needs to learn to respond 
	    # to it.
            V[T-1, b, :]=0
            V[T-1, b, 6]=1

	    # In the original LSTM paper the first symbol was special
	    # in some way. We do the same here for consistency. 
            V[0, b, :]=0
            V[0, b, 7]=1


            M[T, b, 0] = 1
            O[T, b, target] = 1

	    # finally, choose the positions:
            I = randint(self.A_min, self.A_max+1)
            V[I, b, :] = 0 
            V[I, b, A] = 1 

            J = randint(self.B_min, self.B_max+1)
            V[J, b, :] = 0 
            V[J, b, B] = 1 
        return V, O, M









class temporal3(DataAccess):
    def __init__(
        self, 
        T_min, T_max, 
        A_min, A_max, 
        B_min, B_max, 
        C_min, C_max, 

        batch_size, 

        num_train_batches=10,
        num_test_batches=10):


        self.zero_one_loss = zero_one_loss_softmax

        self.out_nonlin = Softmax



        self.T_min = T_min;        self.T_max = T_max;        
        self.A_min = A_min;        self.A_max = A_max;
        self.B_min = B_min;        self.B_max = B_max;
        self.C_min = C_min;        self.C_max = C_max;


        self.batch_size = batch_size
        
        self.V = 8
        self.O = 8 # now there are 8 outputs, representing
		   # (X,X,X),(X,X,Y),(X,Y,X),...,(Y,Y,Y).



        self.T = T_max



        self.train_mem = dict()
        self.test_mem = dict()

        self.batch_fn = self
        
        self.train_batches = [i for i in range(num_train_batches)]
        self.test_batches = [-i-1 for i in range(num_test_batches)]


	
        self.num_outputs = 8
        ### self.num_actual_inputs = 1


    def make_V_O_M(self):

        from numpy import zeros
        V = zeros((self.T_max, self.batch_size, self.v))
        O = zeros((self.T_max, self.batch_size, self.o))
        M = zeros((self.T_max, self.batch_size, 1))



	for b in range(self.batch_size):
		
            from pylab import randint
            A = 4+(b%2) #randint(4,6)
            B = 4+(b/2%2) #randint(4,6)
            C = 4+(b/4%2)

            target = 4*(C-4) + 2*(B-4) + (A-4)

            from numpy.random import randint
            T = randint(self.T_min, self.T_max)

            # 1: generate the random inputs
            V[:, b, :] = expand(randint(0, 4, size=self.T_max), 8)


            V[0, b, :]=0
            V[0, b, 7]=1 # as before, 7 is a symbol used at the beginnig.

            V[T, b, :]=0
            V[T, b, 6]=1 # the prediction time. 

            M[T, b, :] = 1 
            O[T, b, target] = 1

            I = randint(self.A_min, self.A_max+1)
            V[I, b, :] = 0 
            V[I, b, A] = 1 
            
            J = randint(self.B_min, self.B_max+1)
            V[J, b, :] = 0 
            V[J, b, B] = 1 

            K = randint(self.C_min, self.C_max+1)
            V[K, b, :] = 0 
            V[K, b, C] = 1 



        return V, O, M





    







class perm_mem(DataAccess):
    """
The random permutation problem. See the paper. Briefly, 
we've got 100 symbols and a sequence of length T.
The first timestep shows either symbol 1 or 2. The next T-1
timesteps show a random permutation (actually, the name
random permutation is incorrect. The next timesteps simply
show random symbols from 3 to 100 for each remaining timestep,
and the learning objective is to predict the next timestep at -each- timestep. 

"""


    def __init__(
        self, 
        T,



        batch_size, 

        size,

        rand_perm = True,

        num_train_batches=10,
        num_test_batches=10):

        self.size = size
        self.zero_one_loss = zero_one_loss_softmax_last
        # notice how in this problem, when it comes to zero-
        # one loss classification, we care only about the
        # last bit. 


        self.out_nonlin = Softmax


        # we can turn off the random permutation, in which case
        # the objective is to predict the next timestep in the sequence
        # x 2 3 4 5 ... 99 x, where x \in {0,1}. 
        # This problem turns out to be easier for reasons I don't fully
        # understand. 
        self.rand_perm = rand_perm
        if self.rand_perm is False:
            assert batch_size == 2

        self.batch_size = batch_size
        
        self.V = size
        self.O = size

        self.T = T

        self.train_mem = dict()
        self.test_mem = dict()

        self.batch_fn = self

        self.train_batches = [i for i in range(num_train_batches)]
        self.test_batches = [-i-1 for i in range(num_test_batches)]

        self.num_outputs = T # I used that one. 
        # would've probably been better to ues T * size. 
        ### self.num_actual_inputs = 1


    def make_V_O_M(self):
        from numpy import zeros, ones
        V = zeros((self.T, self.batch_size, self.V))
        O = zeros((self.T, self.batch_size, self.O))
        M = ones((self.T, self.batch_size, 1))

        from pylab import randint, arange, permutation

	for b in range(self.batch_size):
            
            if self.rand_perm:
                # choose the random elements of the sequence
                bz = randint(2, self.size, 
                             size=self.T-1)
                # choose the first bit to memorize (the "x" in before)
                a = b % 2 # again, use b%2 to balance the training cases.

            else:
                # here bz is the constant array (which we define
                # to wrap-around)
                bz = 2 + (arange(2, self.T-1) % (self.size-2))
                # and the target.
                a = b % 2
            

            V[0, b, a]=1
            V[1:, b, :] = expand(bz, self.size)

            O[-1, b, a]=1
            O[:-1, b, :] = expand(bz, self.size)

            # note that M = 1 (above)

        return V, O, M







class raw_mem(DataAccess):
    """
    The raw_mem class defines the 5 and the 20 bit noiseless memorization
    problems. It can also define harder memorization problems. Have a look inside!

    """
    def __init__(self, 
                 T,
                 batch_size,

                 num_chars, # how many characetrs to memorize
                 char_size, # how large each character is (ie {0,1} or {0,..,4} etc)

                 teacher_force=False, 
                 # teacher_force is a way of making the problem easier.
                 # If the RNN needs to predict x_t,x_{t+1},...,x_{t+k},
                 # then it will receive x_{t-1},...,x_{t+k-1} as inputs.
                 # In other words, once the RNN makes a prediction, it 
                 # gets to see the taregt. 


                 only_predict_the_memory_bits=False,
                 # if True, we predict only the last num_chars timesteps.
                 # and don't care about the net's behaviour in the rest of the 
                 # timesteps.
                 # If false, the RNN needs to predict the entire sequence
                 # which is basically a third symbol except in the last 
                 # 

                 num_train_batches=10,
                 num_test_batches=10):

        self.teacher_force = teacher_force
        self.num_chars = num_chars
        self.char_size = char_size

        self.zero_one_loss = zero_one_loss_softmax

        self.out_nonlin = Softmax

        self.T = T

        self.batch_size = batch_size
        
        self.V = char_size
        self.O = char_size


        self.T = T

        self.only_predict_the_memory_bits = only_predict_the_memory_bits


        self.train_mem = dict()
        self.test_mem = dict()

        self.batch_fn = self
        
        self.train_batches = [i for i in range(num_train_batches)]
        self.test_batches = [-i-1 for i in range(num_test_batches)]

	
        if only_predict_the_memory_bits:
            self.num_outputs = char_size * num_chars
        else:
            # this isn't quite the right number of outputs, but that's what
            # I used in the RNN experiment. I'm pretty sure the right number
            # of output units will work also. 
            self.num_outputs = char_size * num_chars * T


        ### self.num_actual_inputs = 1



    def make_V_O_M(self):
        from numpy import zeros, ones
        V = zeros((self.T, self.batch_size, self.V))
        O = zeros((self.T, self.batch_size, self.O))
        M = zeros((self.T, self.batch_size, 1))

        from pylab import rand, randn

	for b in range(self.batch_size):
            from pylab import randint

            if self.batch_size < self.char_size**self.num_chars:
                # our minibatch is too small and requires random
                # sampling
                data = randint(self.char_size, size=self.num_chars)

            elif self.batch_size == self.char_size**self.num_chars:
                # our minibatch is large enoguh to contain all possible
                # sequences, so each possible sequence will appear precisely
                # once in the batch.
                from pylab import base_repr, amap
                data = amap(int, 
                         base_repr(
                            b, 
                            base=self.char_size, 
                            padding=self.num_chars))[-self.num_chars:]
            else:
                raise TypeError("self.batch_size (%s) is bigger than %s" % 
                                (self.batch_size, self.char_size**self.num_chars))

                
            # input
            V[:, b, -1] = 1

            V[:self.num_chars, b, :] = 0
            V[:self.num_chars, b, :self.char_size] = \
                expand(data, 
                       self.char_size)


            if self.teacher_force:
                # I'll admit I haven't tested teacher_force.
                V[-self.num_chars:, b, :] = 0
                V[-self.num_chars:, b, :self.char_size] = \
                    expand(data, 
                           self.char_size)


            # a pre signal. Cool. 
            V[-self.num_chars-1, b, :] = 0
            V[-self.num_chars-1, b, -2] = 1

            
            # output: 
            O[:, b, -1] = 1
            O[-self.num_chars:, b, :] = 0
            O[-self.num_chars:, b, :self.char_size] = \
                expand(data, 
                       self.char_size)

            if self.only_predict_the_memory_bits:
                M[:, b, :] = 0
                M[-self.num_chars:, b, :] = 1
            else:
                M[:, b, :] = 1

        return V, O, M


    
