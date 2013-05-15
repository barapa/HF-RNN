import npmat as npmat
import gnumpy
import gnumpy as g
import gnumpy_cpu as gc
import gnumpy_conv_tmp as conv

reload (conv)
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


def test_conv(plot=False, 
              g=1,
              i=2,
              f=2,
               color = 0,
               iorder = 1,
               image_size = 8,
               filter_size = 4,
               num_groups = 4,
               num_images_per_group = 4,
               num_filters_per_group = 2):
    # use the same data for both experimentps:

    from pylab import randn

    output_size = image_size - filter_size + 1

    num_filters_per_group = 40

    color_mult = [1,3][color]


    images = rand(num_groups,  num_images_per_group, color_mult, image_size, image_size) < 0.1
    filters = rand(num_groups, num_filters_per_group, color_mult, filter_size, filter_size)< 0.5
    filters[:,1,:,:,:]=1

    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([gnumpy, gc]): 
        conv.g = gx 
        conv._cu = gx._cudamat
        
        # I don't get it. And I really need to get if I'm to apply it anywhere at all.
        # How are these things working? 
        ans[i] = conv.conv(gx.garray(images),
                           gx.garray(filters),
                           iorder=iorder).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    # quite successful, indeed.  It's weird as hell. The strangest
    # thing is that it used to work.

    if plot:

        from pylab import show, subplot
        c=[0,1][color]

#enum ORDER {
#    GROUP_FILTER_IMAGE, IMAGE_GROUP_FILTER
 #GFI                         # IGF
        if iorder==1:
            print 'iorder=1, yes.'
            #                         g,i,c,
            subplot(331); show(images[g,i,c,:,:]) 
            subplot(332); show(filters[g,f,c,:,:])

            #                         i,g,f
            print 'f=',f
            print ans[1].shape
            subplot(334); show(ans[0][i,g,f,:,:])
            subplot(335); show(ans[1][i,g,f,:,:])
        else:
            subplot(331); show(images[g,i,c,:,:]) 
            subplot(332); show(filters[g,f,c,:,:])

            #                         g,f,i
            subplot(334); show(ans[0][g,f,i,:,:])
            subplot(335); show(ans[1][g,f,i,:,:])



    return ans




def test_conv2(plot=False, 
               color = 0,
               iorder = 1,
               image_size = 17,
               filter_size = 5,
               num_groups = 8,
               num_images_per_group = 4,
               num_filters_per_group = 10,
               g=1,
               i=2,
               c=0,
               f=2,
               p = .1
               ):
    # use the same data for both experimentps:

    from pylab import randn

    output_size = image_size - filter_size + 1

    num_filters_per_group = 40

    color_mult = [1,3][color]



    images = rand(num_groups, num_images_per_group, color_mult, image_size, image_size) < p
    if iorder==1:
        filters = rand(num_images_per_group, num_groups, num_filters_per_group, output_size, output_size)< p
        filters[:,:,0]=1
    else:
        filters = rand(num_groups, num_filters_per_group, num_images_per_group, output_size, output_size)< p
        filters[:,0,:]=1

    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([gnumpy, gc]): 
        conv.g = gx 
        conv._cu = gx._cudamat
        
        # I don't get it. And I really need to get if I'm to apply it anywhere at all.
        # How are these things working? 
        ans[i] = conv.conv2(gx.garray(images),
                            gx.garray(filters),
                            targets = None,
                            iorder=iorder).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    # quite successful, indeed.  It's weird as hell. The strangest
    # thing is that it used to work.
    if c is None:
        c=[0,1][color]
    if color==0:
        c=0

    if plot:
        from pylab import show, subplot

        if iorder==1:

            #                         g,i,c,
            subplot(331); show(images[g,i,c,:,:]) 
            #             iorder==1:  igf
            subplot(332); show(filters[i,g,f,:,:])

            #                         g,i,f     
            subplot(334); show(ans[0][i,g,f,c,:,:])
            subplot(335); show(ans[1][i,g,f,c,:,:])
        else:
            subplot(331); show(images[g,i,c,:,:]) 
            subplot(332); show(filters[g,f,i,:,:])

            subplot(334); show(ans[0][i,g,f,c,:,:])
            subplot(335); show(ans[1][i,g,f,c,:,:])



    return ans



def test_conv3(plot=False, 
               color = 0,
               iorder = 1,
               image_size = 17,
               filter_size = 3,
               num_groups = 4,
               num_images_per_group = 10,
               num_filters_per_group = 4,
               i=1,
               f=2,
               g=3,
               c=1,
               p=.1):
    # use the same data for both experimentps:

    from pylab import randn

    color_mult = [1,3][color]

    if iorder==0:
        images = rand(num_groups, num_filters_per_group, num_images_per_group, image_size, image_size)
    elif iorder==1:
        images = rand(num_images_per_group, num_groups, num_filters_per_group, image_size, image_size)

    images = images < p

    filters = randn(num_groups, num_filters_per_group, color_mult, filter_size, filter_size)
    filters[:,0]=1

    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([gnumpy, gc]): 
        conv.g = gx 
        conv._cu = gx._cudamat

        ans[i] = conv.conv3(gx.garray(images),
                            gx.garray(filters),
                            targets = None,
                            iorder=iorder).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    if color==0 and c!= 0:
        print 'color=0, so setting c to 0'
        c=0

    if plot:
        from pylab import show, subplot
        if iorder==1:
            #                         i g f
            subplot(221); show(images[i,g,f,:,:]) 
            #                         g i c  
            subplot(223); show(ans[0][g,i,c,:,:])
            subplot(224); show(ans[1][g,i,c,:,:])
        else:
            subplot(221); show(images[g,f,i,:,:]) 

            #                         i g f
            subplot(223); show(ans[0][g,i,c,:,:])
            subplot(224); show(ans[1][g,i,c,:,:])


    return ans


def test_battery():
    test_conv(iorder=0, color=0)
    test_conv(iorder=1, color=0)
    test_conv(iorder=0, color=1)
    test_conv(iorder=1, color=1)

    test_conv2(iorder=0, color=0)
    test_conv2(iorder=1, color=0)
    test_conv2(iorder=0, color=1)
    test_conv2(iorder=1, color=1)

    test_conv3(iorder=0, color=0)
    test_conv3(iorder=1, color=0)
    test_conv3(iorder=0, color=1)
    test_conv3(iorder=1, color=1)
