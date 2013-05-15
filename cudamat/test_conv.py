

def test_rot180():
    import cudamat
    import numpy as np
    num_images = 100
    img_size = 50
    img_tot_size = 50*50


    inputs = np.random.randn(num_images, img_tot_size)
    inputs[1:] = (np.random.rand(*inputs[1:].shape)<0.5)
    inputs[0] = (np.random.rand(*inputs[1].shape)<0.005)

    targets = np.random.randn(num_images, img_tot_size)

    cu_inputs = cudamat.CUDAMatrix(inputs.T)
    cu_targets = cudamat.CUDAMatrix(targets.T)

    cudamat._cudamat.rot180(cu_inputs.p_mat, cu_targets.p_mat, 0)

    cua_targets = cu_targets.asarray().T

    targets = np.array([x[::-1,::-1]
                        for x in inputs.reshape(num_images, img_size, img_size)]).reshape(num_images, img_tot_size)

    print abs(targets - cua_targets).max()

    from pylab import imshow, subplot, gray
    gray()
    subplot(221)
    imshow(inputs[0].reshape(img_size, img_size), interpolation='nearest')
    subplot(222)
    imshow(targets[0].reshape(img_size, img_size), interpolation='nearest')

    subplot(223)
    imshow(cua_targets[0].reshape(img_size, img_size), interpolation='nearest')



def test_copyInto():
    import cudamat
    import numpy as np
    num_images = 100
    img_size = 50
    target_size = 72
    img_tot_size = img_size**2
    target_tot_size = target_size**2

    inputs = np.random.randn(img_tot_size, num_images)
    targets = np.zeros((target_tot_size, num_images))

    cu_inputs = cudamat.CUDAMatrix(inputs)
    cu_targets = cudamat.CUDAMatrix(targets)

    assert (target_size - img_size) % 2 == 0 and padding2>=0
    padding = (target_size - img_size)/2 
    cudamat._cudamat.copy_into_center(cu_inputs.p_mat, cu_targets.p_mat, padding, 0)

    cua_targets = cu_targets.asarray()

    print abs(targets - cua_targets).max()

    from pylab import imshow, subplot, gray
    gray()

    #subplot(221)
    #imshow(inputs[0].reshape(img_size, img_size), interpolation='nearest')
    subplot(222)
    imshow(inputs[:,0].reshape(img_size, img_size), interpolation='nearest')

    subplot(223)
    imshow(cua_targets[:,0].reshape(target_size, target_size), interpolation='nearest')





def test_copyOutOf():
    import cudamat
    import numpy as np
    num_images = 100
    img_size = 50
    target_size = 72
    img_tot_size = img_size**2
    target_tot_size = target_size**2

    targets = np.random.randn(target_tot_size, num_images)<-2
    inputs = np.zeros((img_tot_size, num_images))

    cu_inputs = cudamat.CUDAMatrix(inputs)
    cu_targets = cudamat.CUDAMatrix(targets)

    assert (target_size - img_size) % 2 == 0
    padding = (target_size - img_size)/2
    cudamat._cudamat.copy_out_of_center(cu_targets.p_mat, cu_inputs.p_mat, padding, 0)

    cua_inputs = cu_inputs.asarray()

    #print abs(targets - cua_targets).max()

    from pylab import imshow, subplot, gray
    gray()

    #subplot(221)
    #imshow(inputs[0].reshape(img_size, img_size), interpolation='nearest')
    subplot(222)
    imshow(targets[:,1].reshape(target_size, target_size), interpolation='nearest')

    subplot(223)
    imshow(cua_inputs[:,1].reshape(img_size, img_size), interpolation='nearest')





def test_grid_to_matrix():
    import cudamat
    import numpy as np

    img_size = 40
    square_size = 10

    num_images = 128*96
    img_pixels = img_size * img_size
    regions_per_image = (img_size / square_size) ** 2

    

    inputs = np.random.rand(img_pixels, num_images)
    targets = np.random.rand(square_size * square_size, num_images * regions_per_image)
    targets2 = inputs*0    

    # apply the cudat.
    cu_inputs = cudamat.CUDAMatrix(inputs)
    cu_targets = cudamat.CUDAMatrix(targets)
    cu_targets2 = cudamat.CUDAMatrix(targets2)

    cudamat._cudamat.grid_to_matrix(cu_inputs.p_mat, cu_targets.p_mat, square_size) 
    cudamat._cudamat.matrix_to_grid(cu_targets.p_mat, cu_targets2.p_mat, square_size)

    print 'done dealing with the GPU.'


    from pylab import subplot, imshow, gray
    gray()
    subplot(141)
    imshow(cu_inputs.asarray()[:,0].reshape(img_size, img_size), interpolation='nearest')

    subplot(142)
    imshow(cu_targets.asarray()[:,0].reshape(square_size, square_size).T, interpolation='nearest')

    subplot(143)
    imshow(cu_targets2.asarray()[:,0].reshape(img_size, img_size), interpolation='nearest')


