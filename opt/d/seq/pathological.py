import opt.d.seq.utils as u
import opt.utils.nonlin as n
import gnumpy as g
import numpy as np

"""
This script defines a numebr of sample problems. Their names 
should be self-explanatory. 

Mess with them.
Modify them. Try T_min = 1000 and T_max = 1100 and see how badly
it fails. (Though it can sometimes succeed on the addition problem
and on the noisy memorization problems.)


batch_size is a global parameter that affects all problems 
except the tiny memorization (5 bits), where batch_size is 32. 
I found that a batch_size of 300 can solve the T=100 versions,
but to solve the T=200, you'd 
"""

# feel free to play with these quantities. batch_size = 1000
# is more reliably than smaller ones.
batch_size = 1000


T = 200

factor = T/100.
f = lambda x: int(x*factor)


add    = u.op_cls(T_min = int(100*factor), T_max = int(110*factor),

                  A_min = f(1),
                  A_max = f(10),

                  B_min = f(10),
                  B_max = f(50),

                  op = lambda x,y: x+y,

                  inp_dist = lambda sh: np.random.rand(sh),
                  # notice the use of rand over randn.  with rand, the
                  # numbers are all positive, which makes the
                  # multiplication problem much, much easier. If
                  # negatvie numbers are allowed, then solving
                  # multiplication is harder than solving the Xor 
                  # problem, which we already know to be hard for RNNs.

                  batch_size = batch_size,


                  num_train_batches = 10,
                  num_test_batches = 10,

                  out_nonlin = n.Lin, # we minimize the squared loss
                  
                  zero_one_loss = u.zero_one_loss_real_valued)


mult   = u.op_cls(T_min = f(100), T_max = f(110),

                  A_min = f(1),
                  A_max = f(10),

                  B_min = f(10),
                  B_max = f(50),

                  op = lambda x,y: x*y,


                  inp_dist = lambda sh: np.random.rand(sh),
                  # Note that we use rand and not randn. 
                  # This is the way it it's done in the original
                  # LSTM paper. Using randn instead of rand will
                  # make the problem much more difficult, because
                  # the RNN essentially needs to solve the XOR
                  # problem when restricted to the signs of the
                  # random inputs. 


                  batch_size = batch_size, 

                  num_train_batches = 10,
                  num_test_batches = 10,

                  out_nonlin = n.Lin, 
                  
                  zero_one_loss = u.zero_one_loss_real_valued)




## The Xor problem is harder than the rest. It can still be solve
## with HF at T=100, but success is very rare. Typically the RNNs
## fail to notice any long range dependencies whatsoever.
xor   = u.op_cls(T_min = f(50), T_max = f(55),

               A_min = f(1), A_max = f(10),
               B_min = f(15), B_max = f(25),

               batch_size = batch_size,

               op = lambda x,y: (np.array(x).astype(int) ^ 
                                 np.array(y).astype(int)),

               inp_dist = lambda sh: (np.random.rand(sh) < 0.5).astype(float),


               num_train_batches = 10,
               num_test_batches = 10,

               out_nonlin = n.Sigmoid, # minimize cross-entropy
                  
               
               zero_one_loss = u.zero_one_loss_sigm
               # and compare the sign of the prediction to the
               # target
               ) 


temporal2 = u.temporal2(T_min = f(100), T_max = f(110),
                        A_min = f(1), A_max = f(10),
                        B_min = f(40), B_max = f(50),

                        batch_size = batch_size)


temporal3 = u.temporal3(T_min = f(100), T_max = f(110),

                        A_min = f(1), A_max = f(10),
                        B_min = f(30), B_max = f(40),
                        C_min = f(50), C_max = f(60),

                        batch_size = batch_size)

                        

rand_perm = u.perm_mem(T = f(100), 
                       size = 100, 
                       # the number of symbols used in the
                       # "permutation"

                       batch_size = batch_size)





raw_mem5 = u.raw_mem(T = f(100),
                     batch_size = 32,
                     num_chars = 5,
                     char_size = 2)
                     
raw_mem20 = u.raw_mem(T = f(100),
                     batch_size = batch_size,
                     num_chars = 10,
                     char_size = 5)
# when using these two, don't forget to use a much lower threshold for
# the zero-one-loss for stopping HF, because these problems have an
# ouptut unit per time step.





raw_mem20_only_last_bits = u.raw_mem(T = f(100),
                                     batch_size = batch_size,
                                     num_chars = 10,
                                     char_size = 5,
                                     only_predict_the_memory_bits = True)

raw_mem5_only_last_bits = u.raw_mem(T = f(100),
                                    batch_size = 32,
                                    num_chars = 5,
                                    char_size = 2,
                                    only_predict_the_memory_bits = True)
# these two are different, because they have a small number of output
# units in the end of the sequence.
                     
