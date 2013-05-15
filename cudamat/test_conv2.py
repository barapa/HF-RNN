


import npmat as npmat
import gnumpy
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


    
def test_rot180_l(plot=False, color=1):
    # use the same data for both experiments:
    num_imgs = 200

    color_mult = [1,3][color]

    img_pix=(10**2)*color_mult
    inputs = rand(num_imgs,img_pix)<.1
    inputs[1:]=0

    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g,    gc]):
        conv.g = gx # wonderful. This seems to work. At least. 
        conv._cu = gx._cudamat
        
        a = gx.garray(inputs)
        ans[i] = conv.rot180_l(a, color=color).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    if plot:
        if not color:
            from pylab import show, subplot
            subplot(221); show(inputs[0])
            subplot(223); show(ans[0][0])
            subplot(224); show(ans[1][0])
        else:
            from pylab import show, subplot
            r = img_pix/color_mult
            subplot(331); show(inputs[0][:r])
            subplot(332); show(ans[0][0][:r])

            subplot(334); show(ans[1][0][:r])
            subplot(335); show(ans[1][0][r:2*r])
            subplot(336); show(ans[1][0][2*r:3*r])



def test_rot180(plot=False, color=1):
    # use the same data for both experiments:
    num_imgs = 200

    color_mult = [1,3][color]

    image_size = 10
    inputs = rand(num_imgs,color_mult,image_size,image_size)<.1
    inputs[1:]=0

    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g,    gc]):
        conv.g = gx # wonderful. This seems to work. At least. 
        conv._cu = gx._cudamat
        
        a = gx.garray(inputs)
        ans[i] = conv.rot180(a).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    if plot:
        if not color:
            from pylab import show, subplot
            subplot(221); show(inputs[0])
            subplot(223); show(ans[0][0])
            subplot(224); show(ans[1][0])
        else:
            from pylab import show, subplot
            r = img_pix/color_mult
            subplot(331); show(inputs[0][:r])
            subplot(332); show(ans[0][0][:r])

            subplot(334); show(ans[1][0][:r])
            subplot(335); show(ans[1][0][r:2*r])
            subplot(336); show(ans[1][0][2*r:3*r])


def test_copy_into_center_l(plot=False, color=1):
    # use the same data for both experiments:
    num_imgs = 200

    color_mult = [1,3][color]

    img_res = 15
    padding = 4

    img_pix=img_res**2*color_mult

    inputs = rand(num_imgs,img_pix)<.1
    inputs[1:]*=0 

    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g,    gc]):
        conv.g = gx # wonderful. This seems to work. At least. 
        conv._cu = gx._cudamat
        
        a = gx.garray(inputs)
        ans[i] = conv.copy_into_center_l(a, padding=padding, color=color).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    if plot:
        if not color:
            from pylab import show, subplot
            subplot(221); show(inputs[0])
            subplot(223); show(ans[0][0])
            subplot(224); show(ans[1][0])
        else:
            from pylab import show, subplot
            r = img_pix/color_mult
            subplot(331); show(inputs[0][:r])
            subplot(332); show(ans[0][0][:r])

            r = ans[0].shape[1]/3
            subplot(334); show(ans[1][0][:r])
            subplot(335); show(ans[1][0][r:2*r])
            subplot(336); show(ans[1][0][2*r:3*r])



def test_copy_into_center(plot=False, color=1):
    # use the same data for both experiments:
    num_imgs = 200

    color_mult = [1,3][color]

    img_res = 15
    padding = 4

    img_pix=img_res**2*color_mult

    inputs = rand(num_imgs,color_mult,img_res,img_res)<.1
    inputs[1:]*=0 

    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g,    gc]):
        conv.g = gx # wonderful. This seems to work. At least. 
        conv._cu = gx._cudamat
        
        a = gx.garray(inputs)
        ans[i] = conv.copy_into_center(a, padding).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    if plot:
        if not color:
            from pylab import show, subplot
            subplot(221); show(inputs[0])
            subplot(223); show(ans[0][0])
            subplot(224); show(ans[1][0])
        else:
            from pylab import show, subplot
            r = img_pix/color_mult
            subplot(331); show(inputs[0][:r])
            subplot(332); show(ans[0][0][:r])

            r = ans[0].shape[1]/3
            subplot(334); show(ans[1][0][:r])
            subplot(335); show(ans[1][0][r:2*r])
            subplot(336); show(ans[1][0][2*r:3*r])





def test_add_into_center_l(plot=False, color=1):
    # use the same data for both experiments:
    num_imgs = 200

    color_mult = [1,3][color]

    img_res = 15
    padding = 4

    img_pix=img_res**2*color_mult
    targ_pix=(img_res+2*padding)**2*color_mult

    inputs = rand(num_imgs,img_pix)<.1
    inputs[1:]*=0 

    targets = rand(num_imgs,targ_pix)


    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g,    gc]):
        conv.g = gx # wonderful. This seems to work. At least. 
        conv._cu = gx._cudamat
        
        a = gx.garray(inputs)
        ans[i] = conv.add_into_center_l(a, gx.garray(targets), color=color).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    if plot:
        if not color:
            from pylab import show, subplot
            subplot(221); show(inputs[0])
            subplot(223); show(ans[0][0])
            subplot(224); show(ans[1][0])
        else:
            from pylab import show, subplot
            r = img_pix/color_mult
            subplot(331); show(inputs[0][:r])


            r = ans[0].shape[1]/3
            subplot(332); show(ans[0][0][:r])
            subplot(334); show(ans[1][0][:r])
            subplot(335); show(ans[1][0][r:2*r])
            subplot(336); show(ans[1][0][2*r:3*r])






def test_add_into_center(plot=False, color=1):
    # use the same data for both experiments:
    num_imgs = 200

    color_mult = [1,3][color]

    img_res = 15
    padding = 4

    img_pix=img_res**2*color_mult

    inputs = rand(num_imgs,color_mult,img_res,img_res)<.1
    inputs[1:]*=0 

    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g,    gc]):
        conv.g = gx # wonderful. This seems to work. At least. 
        conv._cu = gx._cudamat
        
        a = gx.garray(inputs)
        ans[i] = conv.add_into_center(a, padding).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    if plot:
        if not color:
            from pylab import show, subplot
            subplot(221); show(inputs[0])
            subplot(223); show(ans[0][0])
            subplot(224); show(ans[1][0])
        else:
            from pylab import show, subplot
            r = img_pix/color_mult
            subplot(331); show(inputs[0][:r])
            subplot(332); show(ans[0][0][:r])

            r = ans[0].shape[1]/3
            subplot(334); show(ans[1][0][:r])
            subplot(335); show(ans[1][0][r:2*r])
            subplot(336); show(ans[1][0][2*r:3*r])







def test_copy_out_of_center_l(plot=False, color=1):
    # use the same data for both experiments:
    num_imgs = 200

    color_mult = [1,3][color]

    img_res = 15
    padding = 4

    targ_pix=img_res**2*color_mult
    img_pix=(img_res+2*padding)**2*color_mult

    inputs = rand(num_imgs,img_pix)<.1
    inputs[1:]*=0 

    #targets = g.rand(num_imgs,targ_pix)


    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g,    gc]):
        conv.g = gx # wonderful. This seems to work. At least. 
        conv._cu = gx._cudamat
        
        a = gx.garray(inputs)
        ans[i] = conv.copy_out_of_center_l(a, padding=4, color=color).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    if plot:
        if not color:
            from pylab import show, subplot
            subplot(221); show(inputs[0])
            subplot(223); show(ans[0][0])
            subplot(224); show(ans[1][0])
        else:
            from pylab import show, subplot
            r = img_pix/color_mult
            subplot(331); show(inputs[0][:r])
            subplot(332); show(ans[0][0][:r])

            r = ans[0].shape[1]/3
            subplot(334); show(ans[1][0][:r])
            subplot(335); show(ans[1][0][r:2*r])
            subplot(336); show(ans[1][0][2*r:3*r])





def test_copy_out_of_center(plot=False, color=1):
    # use the same data for both experiments:
    num_imgs = 200

    color_mult = [1,3][color]

    img_res = 15
    padding = 4

    img_pix=img_res**2*color_mult

    inputs = rand(num_imgs,color_mult,img_res,img_res)<.1
    inputs[1:]*=0 

    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g,    gc]):
        conv.g = gx # wonderful. This seems to work. At least. 
        conv._cu = gx._cudamat
        
        a = gx.garray(inputs)
        ans[i] = conv.copy_out_of_center(a, padding).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    if plot:
        if not color:
            from pylab import show, subplot
            subplot(221); show(inputs[0])
            subplot(223); show(ans[0][0])
            subplot(224); show(ans[1][0])
        else:
            from pylab import show, subplot
            r = img_pix/color_mult
            subplot(331); show(inputs[0][:r])
            subplot(332); show(ans[0][0][:r])

            r = ans[0].shape[1]/3
            subplot(334); show(ans[1][0][:r])
            subplot(335); show(ans[1][0][r:2*r])
            subplot(336); show(ans[1][0][2*r:3*r])









def test_add_out_of_center_l(plot=False, color=1):
    # use the same data for both experiments:
    num_imgs = 200

    color_mult = [1,3][color]

    img_res = 15
    padding = 4

    targ_pix=img_res**2*color_mult
    img_pix=(img_res+2*padding)**2*color_mult

    inputs = rand(num_imgs,img_pix)<.1
    targets = rand(num_imgs,targ_pix)*0.4

    inputs[1:]*=0 

    #targets = g.rand(num_imgs,targ_pix)


    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g,    gc]):
        conv.g = gx # wonderful. This seems to work. At least. 
        conv._cu = gx._cudamat
        
        a = gx.garray(inputs)
        ans[i] = conv.add_out_of_center_l(a, gx.garray(targets), color=color).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    if plot:
        if not color:
            from pylab import show, subplot
            subplot(221); show(inputs[0])
            subplot(223); show(ans[0][0])
            subplot(224); show(ans[1][0])
        else:
            from pylab import show, subplot
            r = img_pix/color_mult
            subplot(331); show(inputs[0][:r])


            r = ans[0].shape[1]/3
            subplot(332); show(ans[0][0][:r])
            subplot(334); show(ans[1][0][:r])
            subplot(335); show(ans[1][0][r:2*r])
            subplot(336); show(ans[1][0][2*r:3*r])



def test_add_out_of_center(plot=False, color=1):
    # use the same data for both experiments:
    num_imgs = 200

    color_mult = [1,3][color]

    img_res = 15
    padding = 4

    img_pix=img_res**2*color_mult

    inputs = rand(num_imgs,color_mult,img_res,img_res)<.1
    inputs[1:]*=0 

    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g,    gc]):
        conv.g = gx # wonderful. This seems to work. At least. 
        conv._cu = gx._cudamat
        
        a = gx.garray(inputs)
        ans[i] = conv.add_out_of_center(a, padding).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    if plot:
        if not color:
            from pylab import show, subplot
            subplot(221); show(inputs[0])
            subplot(223); show(ans[0][0])
            subplot(224); show(ans[1][0])
        else:
            from pylab import show, subplot
            r = img_pix/color_mult
            subplot(331); show(inputs[0][:r])
            subplot(332); show(ans[0][0][:r])

            r = ans[0].shape[1]/3
            subplot(334); show(ans[1][0][:r])
            subplot(335); show(ans[1][0][r:2*r])
            subplot(336); show(ans[1][0][2*r:3*r])







# that's our downsample.
def test_sub_sample(plot=False, color=1):
    # use the same data for both experiments:
    num_imgs = 200

    # no color-mult. That's cool. Right. Awesome and shit. Don't give up ever.
    # NVMatrix. Cool shit. I like his code. It's very logically clear.

    color=0

    img_res = 40
    factor = 2

    img_pix=img_res**2
    inputs = rand(num_imgs,img_pix)<0.01


    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g,    gc]):
        conv.g = gx # wonderful. This seems to work. At least. 
        conv._cu = gx._cudamat
        
        a = gx.garray(inputs)
        ans[i] = conv.sub_sample(a, factor=factor).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    if plot:
        if not color:
            from pylab import show, subplot
            subplot(221); show(inputs[0])
            subplot(223); show(ans[0][0])
            subplot(224); show(ans[1][0])
        else:
            from pylab import show, subplot
            r = img_pix/color_mult
            subplot(331); show(inputs[0][:r])


            r = ans[0].shape[1]/3
            subplot(332); show(ans[0][0][:r])
            subplot(334); show(ans[1][0][:r])
            subplot(335); show(ans[1][0][r:2*r])
            subplot(336); show(ans[1][0][2*r:3*r])



# that's our downsample.
def test_super_sample(plot=False, color=1):
    # use the same data for both experiments:
    num_imgs = 200

    # no color-mult. That's cool. Right. Awesome and shit. Don't give up ever.
    # NVMatrix. Cool shit. I like his code. It's very logically clear.

    color=0

    img_res = 10
    factor = 2

    img_pix=img_res**2
    inputs = rand(num_imgs,img_pix)<.1


    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g,    gc]):
        conv.g = gx # wonderful. This seems to work. At least. 
        conv._cu = gx._cudamat
        
        a = gx.garray(inputs)
        ans[i] = conv.super_sample(a, factor=factor).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    if plot:
        if not color:
            from pylab import show, subplot
            subplot(221); show(inputs[0])
            subplot(223); show(ans[0][0])
            subplot(224); show(ans[1][0])
        else:
            from pylab import show, subplot
            r = img_pix/color_mult
            subplot(331); show(inputs[0][:r])


            r = ans[0].shape[1]/3
            subplot(332); show(ans[0][0][:r])
            subplot(334); show(ans[1][0][:r])
            subplot(335); show(ans[1][0][r:2*r])
            subplot(336); show(ans[1][0][2*r:3*r])



# that's our downsample.
def test_grid_to_matrix(plot=False):
    # use the same data for both experiments:
    num_imgs = 200

    img_res = 20
    square_size = 10

    img_pix=img_res**2
    inputs = rand(num_imgs,img_pix)<.1


    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g,    gc]):
        conv.g = gx # wonderful. This seems to work. At least. 
        conv._cu = gx._cudamat
        
        a = gx.garray(inputs)
        ans[i] = conv.grid_to_matrix(a, square_size).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    
    # quite successful, indeed. 

    if plot:
        from pylab import show, subplot
        subplot(221); show(inputs[0])
        subplot(223); show(ans[0][0])
        subplot(224); show(ans[1][0])



def test_matrix_to_grid(plot=False):
    # use the same data for both experiments:
    num_imgs = 200

    img_res = 20
    square_size = 10

    regions_per_square = (img_res/square_size)**2

    img_pix=img_res**2
    inputs = rand(num_imgs*regions_per_square,square_size**2)<.1
    
    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g,    gc]):
        conv.g = gx # wonderful. This seems to work. At least. 
        conv._cu = gx._cudamat
        
        a = gx.garray(inputs)
        ans[i] = conv.matrix_to_grid(a, img_res).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    
    # quite successful, indeed. 

    if plot:
        from pylab import show, subplot
        subplot(221); show(inputs[0])
        subplot(223); show(ans[0][0])
        subplot(224); show(ans[1][0])






#def conv_test():
#    import gnumpy as g
#    imgs = g.rand(101,49) # just do the fucking usual thing. That's all. 
#    filters = g.rand(22, 16)
#    num_groups = 1
#    ans = conv(imgs,filters,num_groups,iorder=1)
#    print ans.shape




def test_conv_l(plot=False):
    # use the same data for both experiments:

    from pylab import randn

    color = 0
    image_size = img_size = 10
    filter_size = 5 
    num_groups = 6
    num_images_per_group = 2
    num_filters_per_group = 4
    iorder = 1

    color_mult = [1,3][color]

    images_pixels  = (image_size**2)*color_mult
    filters_pixels = (filter_size**2)*color_mult
    num_outputs = (img_size - filter_size + 1)**2

    images = randn(num_images_per_group * num_groups, images_pixels)
    filters = randn(num_filters_per_group * num_groups, filters_pixels)

    from pylab import zeros
    if iorder==1:
        targets = zeros((num_images_per_group *  num_groups,
                         num_filters_per_group * num_outputs))
    elif iorder==0:
        targets = zeros((num_filters_per_group *  num_groups,
                         num_images_per_group * num_outputs))
    else: raise TypeError


    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g, gc]): 
        conv.g = gx 
        conv._cu = gx._cudamat
        
        # I don't get it. And I really need to get if I'm to apply it anywhere at all.
        # How are these things working? 
        ans[i] = conv.conv_l(gx.garray(images),
                           gx.garray(filters),
                           targets = gx.garray(targets),                           
                           num_groups = num_groups,
                           color = color,
                           iorder=iorder).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    # quite successful, indeed.  It's weird as hell. The strangest
    # thing is that it used to work.

    if plot:
        from pylab import show, subplot
        subplot(221); show(images[0])
        subplot(223); show(ans[0][0])
        subplot(224); show(ans[1][0])


    return ans





def test_conv_works_with_npmat(plot=False):
    # use the same data for both experiments:

    from pylab import randn

    color = 0
    image_size = img_size = 10
    filter_size = 2
    num_groups = 2
    num_images_per_group = 3
    num_filters_per_group = 2
    iorder = 1


    color_mult = [1,3][color]

    images = rand(num_groups, num_images_per_group, color_mult, image_size, image_size) < 0.1
    filters = randn(num_groups, num_filters_per_group, color_mult, filter_size, filter_size)
    filters[:,0]=1

    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g, gc]): 
        conv.g = gx 
        conv._cu = gx._cudamat
        
        # I don't get it. And I really need to get if I'm to apply it anywhere at all.
        # How are these things working? 
        ans[i] = conv.conv(gx.garray(images),
                           gx.garray(filters),
                           targets = None,
                           iorder=iorder).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    # quite successful, indeed.  It's weird as hell. The strangest
    # thing is that it used to work.

    if plot:
        from pylab import show, subplot
        subplot(221); show(images[1,1,0,:,:])
        subplot(223); show(ans[0][1,1,0,:,:])
        subplot(224); show(ans[1][1,1,0,:,:])


    return ans








def test_conv(plot=False,     iorder = 1, color=1,
                      g=1, c=None,f=1,i=1):
    # use the same data for both experiments:

    from pylab import randn

    image_size = img_size = 10
    filter_size = 2
    num_groups = 4
    num_images_per_group = 3
    num_filters_per_group = 6



    color_mult = [1,3][color]

    images = rand(num_groups, num_images_per_group, color_mult, image_size, image_size) < 0.02
    filters = randn(num_groups, num_filters_per_group, color_mult, filter_size, filter_size)

    filters[:,0,:,:,:]=1
    if color_mult==3:
        filters[:,:,0,:,:]=0 # to have only one color filter, or else we won't have
        filters[:,:,2,:,:]=0 # pretty responses.


    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([gnumpy, gc]): 
        conv.g = gx 
        conv._cu = gx._cudamat
        
        # I don't get it. And I really need to get if I'm to apply it anywhere at all.
        # How are these things working? 
        ans[i] = conv.conv(gx.garray(images),
                           gx.garray(filters),
                           targets = None,
                           iorder=iorder).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    # quite successful, indeed.  It's weird as hell. The strangest
    # thing is that it used to work.

    if plot:

        if c is None:
            c=[0,1][color]

        from pylab import show, subplot
        if iorder==1:
            subplot(221); show(images[g,i,c,:,:])
            subplot(223); show(ans[0][g,i,f,:,:])
            subplot(224); show(ans[1][g,i,f,:,:])
        else:
            subplot(221); show(images[g,i,c,:,:])
            subplot(223); show(ans[0][g,f,i,:,:])
            subplot(224); show(ans[1][g,f,i,:,:])

        #subplot(224); show(ans[0][1,1,1,:,:])

    return ans



def test_conv2(plot=False, 
               color = 0,
               iorder = 1,
               image_size = 17,
               filter_size = 5,
               num_groups = 8,
               num_images_per_group = 4,
               num_filters_per_group = 10):
    # use the same data for both experimentps:

    from pylab import randn

    output_size = image_size - filter_size + 1

    num_filters_per_group = 40

    color_mult = [1,3][color]


    images = rand(num_groups, num_images_per_group, color_mult, image_size, image_size) < 0.01
    if iorder==1:
        filters = rand(num_groups, num_images_per_group, num_filters_per_group, output_size, output_size)< 0.01
    else:
        filters = rand(num_groups, num_filters_per_group, num_images_per_group, output_size, output_size)<0.01

#date.

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

    if plot:

        from pylab import show, subplot
        g=1
        i=2
        c=[0,1][color]
        f=2

        if iorder==1:

            #                         g,i,c,
            subplot(331); show(images[g,i,c,:,:]) 
            subplot(332); show(filters[g,i,f,:,:])

            #                         g,i,f     
            subplot(334); show(ans[0][g,i,f,c,:,:])
            subplot(335); show(ans[1][g,i,f,c,:,:])
        else:
            subplot(331); show(images[g,i,c,:,:]) 
            subplot(332); show(filters[g,f,i,:,:])

            #                         g,i,f     
            subplot(334); show(ans[0][g,i,f,c,:,:])
            subplot(335); show(ans[1][g,i,f,c,:,:])




        #subplot(224); show(ans[0][1,1,1,:,:])

    return ans



def test_conv3(plot=False, 
               color = 0,
               iorder = 1,
               image_size = 17,
               filter_size = 3,
               num_groups = 4,
               num_images_per_group = 10,
               num_filters_per_group = 4):
    # use the same data for both experimentps:

    from pylab import randn


    color_mult = [1,3][color]

    if iorder==0:
        images = rand(num_groups, num_filters_per_group, num_images_per_group, image_size, image_size)

    elif iorder==1:
        images = rand(num_groups, num_images_per_group, num_filters_per_group, image_size, image_size)

    images = images < .01

    filters = randn(num_groups, num_filters_per_group, color_mult, filter_size, filter_size)


    ans = [None]*2
    print 'a'
    for (i, gx) in enumerate([g, gc]): 
        conv.g = gx 
        conv._cu = gx._cudamat
        
        # I don't get it. And I really need to get if I'm to apply it anywhere at all.
        # How are these things working? 
        ans[i] = conv.conv3(gx.garray(images),
                            gx.garray(filters),
                            targets = None,
                            iorder=iorder).asarray()
        print 'b'

    print abs(ans[0]-ans[1]).max()

    # quite successful, indeed.  It's weird as hell. The strangest
    # thing is that it used to work.

    if plot:

        from pylab import show, subplot
        if iorder==1:
            # what the fuck are we supposed to see here? 

            #                         g i f
            subplot(221); show(images[2,1,0,:,:]) # with iorder=1, this gives me nans.
            #                         g i c 
            subplot(223); show(ans[0][2,1,0,:,:])
            subplot(224); show(ans[1][2,1,0,:,:])
        else:
            #                         g f i
            subplot(221); show(images[1,0,2,:,:]) # with iorder=1, this gives me nans.
            subplot(223); show(ans[0][1,2,0,:,:])
            subplot(224); show(ans[1][1,2,0,:,:])
                                #     g i c


        #subplot(224); show(ans[0][1,1,1,:,:])

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
