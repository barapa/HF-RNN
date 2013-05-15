import cudamat 


import npmat as npmat
import gnumpy as g
import gnumpy_conv as conv
import gnumpy_cpu as gc
reload (conv)
reload (npmat)
from pylab import rand

g.rand(10) # init the gpu version.

def test_gpu_cpu_conversion():
    import time
    input = rand(2000,1000)

    n=100

    for gx in [g, gc]:

        a = gx.garray(input)
        start = time.time()
        for i in range(n):
            b=a+a


        end = time.time()
        print 'total = %f' % ((end-start)/n)


    


