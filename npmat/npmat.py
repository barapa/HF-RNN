

import os, pdb, time, warnings
import numpy as np

__DTYPE__ = np.float128


def dummy():
    return CUDAMatrix(np.zeros((1, 1)))

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    def newFunc(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc

#from cudamat import CUDAMatException
class CUDAMatException(Exception):
    pass



IncompatibleDimensionsException = CUDAMatException("Incompatible matrix dimensions.")

InvalidConfig = CUDAMatException("Invalid Configuration Error (i.e., a dim of the array must be smaller than 2**16.")
## TODO: Figure out which functions produce an invalid config error. These are those who allocate a thread per col/row/elem.
## Those who allocate a bunch of rows per thread, like mult, add, sub, etc, should be immune to the invalid
## configuration error. PS: this error occurs on the real cudamat, which is why it happens. 
## Sum/Max/Cumsum
MAX_DIM = 2**16


class CUDAMatrix(object):
    """
    A CUDAMatrix object represents a matrix of single precision floating point
    numbers on a GPU.
    """

    def __init__(self, array, ref=True):
        if ref:
            self.numpy_array = reformat(array).astype(__DTYPE__)
        else:
            self.numpy_array = array 
        assert self.numpy_array.ndim == 2
        self.trans = False

    def __del__(self):
        pass

    @staticmethod
    def init_random(seed):
        import numpy.random as random
        random.seed(seed)



    @property
    def num_elems(self):
        return self.numpy_array.size

    @property
    def shape(self):
        return self.numpy_array.shape

    def cheap_transpose(self):
        return CUDAMatrix(self.reshape((self.shape[1], self.shape[0])))

    def reshape(self, shape):
        assert shape[0]*shape[1] == self.shape[0]*self.shape[1]
        self.numpy_array.resize(*shape)
        return self

    def copy(self):
        return empty().assign(self)


    def set_np_array(self, X):
        assert X.shape == self.shape
        self.numpy_array[:] = X
        self.copy_to_device()
        return self



    def zero_copy(self):
        return self.copy().assign(0)


    def resize(self, shape):

        if self.shape != shape:

            print 'CUDAMatrix: resize (%s -> %s)' % (self.shape, shape)
            #self.numpy_array = np.resize(self.numpy_array, shape).astype(__DTYPE__)
            self.numpy_array.resize(shape)
            self.numpy_array[:] = 0


        return self
    
    @property
    def T(self):
        return CUDAMatrix(self.numpy_array.T)

    @property
    def mat(self):
        return self.numpy_array


    @deprecated
    def set_shape(self, shape):
        return self.resize(shape)


    def asarray(self):
        """
        Copies the matrix to an ndarray on the CPU and returns it.
        """

        #return reformat(self.numpy_array.copy())
        return self.numpy_array

    def copy_to_device(self):
        """
        Copy the matrix to the GPU.
        """

        pass 



    def select_columns(self, indices, target):
        """
        copies some columns of self into target.
        <indices> must be a row vector. Its elements are float32's representing integers, e.g. "34.0" means the integer "34".
        after this call, for all r,c, target[r,c]=self[r,indices[c]].
        This returns target.
        Negative indices are interpreted in the usual Python way: all elements of <indices> had better be in the range [-self.shape[1], self.shape[1]-1].
        This does bounds checking, but out of bounds indices do not raise an exception (because the programmer was lazy). Instead, they result in NaN values in <target>.
        """

        assert target.shape[0]==self.shape[0]
        assert indices.shape[0]==1 
        assert indices.shape[1] == target.shape[1]

        for c in range(target.shape[1]):
            try:
                target.numpy_array[:,c] = self.numpy_array[:, int(indices.numpy_array.ravel()[c])]
            except IndexError:
                target.numpy_array[:,c] = np.nan
        return target

    def set_selected_columns(self, indices, source):
        """
        copies all columns of source into some columns of self.
        <indices> must be a row vector. Its elements are float32's representing
        integers, e.g. "34.0" means the integer "34". after this call, for all
        r,c, self[r,indices[c]]=source[r,c]. This returns self.
        Negative indices are interpreted in the usual Python way: all elements
        of <indices> had better be in the range [-self.shape[1], self.shape[1]-1].
        This does bounds checking, but out of bounds indices do not raise an
        exception (because the programmer was lazy). Instead, they result in NaN
        values in <self>.
        """

        assert self.shape[0]==source.shape[0]
        assert indices.shape[0]==1
        assert indices.shape[1]==source.shape[1]

        for c in range(source.shape[1]):
            try:
                self.numpy_array[:,int(indices.numpy_array.ravel()[c])] = source.numpy_array[:,c]
            except IndexError:
                self.numpy_array[:,int(indices.numpy_array.ravel()[c])] = np.nan
        return self


    def copy_to_host(self):
        """
        Copy the matrix to the CPU.
        """
        return self.asarray()


    def np(self):
        return self.copy_to_host()




    def assign(self, val):
        """Assign val to self, where val can be a scalar or a CUDAMatrix
        with the same dimensions as self. """


        if isinstance(val, CUDAMatrix):
            self.resize(val.shape)
            self.numpy_array[:] = val.numpy_array


        elif isinstance(val, (int, float, np.float128)):
            self.numpy_array[:] = val

        return self

    def free_device_memory(self):
        """
        Free memory used up by the matrix on the GPU.
        """
        pass


    def set_trans(self, is_trans):
        """
        Set the transposedness flag to is_trans.
        """
        if is_trans is True:
            trans = self.numpy_array.T
            self.numpy_array.resize(trans.shape)
            self.numpy_array[:] = trans
    

    def slice(self, first_col, last_col):
        return CUDAMatrix(self.numpy_array[:, first_col:last_col], ref=False)

    def get_row_slice(self, start, end, target = None):
        """
        Get the rows with indices start through end. If target is not provided
        memory for a new matrix will be allocated.
        """
                        


        ans = CUDAMatrix(self.numpy_array[start:end, :].copy())

        if target is not None:
            target.assign(ans)
        else:
            target = ans

        return target


    def set_row_slice(self, start, end, mat):
        try:
            self.numpy_array[start:end] = mat.numpy_array
        except ValueError:
            raise IncompatibleDimensionsException
        return self


    def get_col_slice(self, start, end, target = None):
        ## NOTE: no .copy()
        ans = self.slice(start, end)

        if target is not None:
            target.assign(ans)
        else:
            target = ans

        return target

    def set_col_slice(self, start, end, mat):
        return self.slice(start, end).assign(mat)





    # def select_columns(self, indices, target):
    #     """
    #     Copies selected columns into a target matrix.
    #     <self>, <indices>, and <target> are all cudamat matrices.
    #     <self> is an M by K matrix.
    #     <indices> is of shape 1 by N. All elements x are expected to be
    #     0<=x<K, and are expected to have nothing after the decimal point (i.e.
    #     to be floats representing integers).
    #     <target> is an M by N matrix that will be filled with the result.
    #     After the operation, for all i,j, target[i, j] = self[i, int(indices[j])]
    #     This returns <target>.
    #     ? idea: No bounds checking is done.
    #     """
    #     M, K = self.shape

    #     one, N = indices.shape
    #     assert one == 1
    #     M_, N_ = target.shape
    #     assert M_ == M and N == N_

    #     np_ints = indices.numpy_array.astype(int)

    #     if not (np_ints.max() < K and np_ints.min() >= 0):
    #         raise ValueError("Index out of bounds.")

        
    #     target.numpy_array[:] = self.numpy_array[:, np_ints.flatten()]



    #     return target
        



    def transpose(self, target = None):

        if target is None:
            return CUDAMatrix(self.numpy_array.T.copy())
        else:
            target.numpy_array.resize((self.shape[1], self.shape[0]))
            target.numpy_array[:] = self.numpy_array.T

        return target


    def assign_transpose(self, t):
        return t.transpose(target = self)



    def fill_with_rand(self):
        """
        Fill matrix on the GPU with random numbers drawn from the uniform
        distribution over the (0,1) interval.
        """
        self.numpy_array[:] = np.random.rand(*self.shape)

        return self





    def fill_with_randn(self):
        """
        Fill matrix on the GPU with random numbers drawn from the standard normal
        distribution.
        """

        self.numpy_array[:] = np.random.randn(*self.shape)

        return self



    def add_col_vec(self, vec, target = None):
        """
        Add vector vec to every column of the matrix. If a target is provided,
        it is used to store the result instead of self.
        """

        a, b = self.shape
        a_, b_ = vec.shape

        if not (b_ == 1 and a_ == a):
            raise IncompatibleDimensionsException


        if target is None:
            target = self

        target.resize(self.shape)

        target.numpy_array[:] = self.numpy_array + vec.numpy_array

        return target

    def assign_add_col_vec(self, a, b):
        return a.add_col_vec(b, target = self)


        
    def add_col_mult(self, vec, mult, target = None):
        """
        Add a multiple of vector vec to every column of the matrix. If a target
        is provided, it is used to store the result instead of self.
        """

        a, b = self.shape
        a_, b_ = vec.shape

        if not (b_ == 1 and a_ == a):
            raise IncompatibleDimensionsException


        if target is None:
            target = self

        target.resize(self.shape)

        target.numpy_array[:] = self.numpy_array + vec.numpy_array * mult

        return target





    def assign_add_col_mult(self, a, m, b):
        return a.add_col_vec(b, m, target = self)


        
    def add_row_vec(self, vec, target = None):
        """
        Add vector vec to every row of the matrix. If a target is provided,
        it is used to store the result instead of self.
        """

        a, b = self.shape
        a_, b_ = vec.shape

        if not (a_ == 1 and b_ == b):
            raise IncompatibleDimensionsException


        if target is None:
            target = self

        target.resize(self.shape)

        target.numpy_array[:] = vec.numpy_array + self.numpy_array

        return target


        
    def assign_add_row_vec(self, a, b):
        return a.add_row_vec(b, target = self)



    def mult_by_col(self, vec, target = None):
        """
        Multiply vector vec into every column of the matrix. If a target is
        provided, it is used to store the result instead of self.
        """


        a, b = self.shape
        a_, b_ = vec.shape

        if not (b_ == 1 and a_ == a):
            raise IncompatibleDimensionsException

        if target is None:
            target = self

        target.resize(self.shape)


        target.numpy_array[:] = vec.numpy_array * self.numpy_array


        return target
        


    def mult_by_row(self, vec, target = None):
        """
        Multiply vector vec into every row of the matrix. If a target is
        provided, it is used to store the result instead of self.
        """

        a, b = self.shape
        a_, b_ = vec.shape

        if not (b_ == b and a_ == 1):
            raise IncompatibleDimensionsException

        if target is None:
            target = self

        target.resize(self.shape)


        target.numpy_array[:] = vec.numpy_array * self.numpy_array

        return target
        




    def sum(self, axis, target = None):
        """
        Sum the matrix along the given dimension, where 0 represents the leading
        dimension and 1 represents the non-leading dimension. If a target is
        not prvided, a new vector is created for storing the result.
        """



        if axis == 0:
            ans = self.numpy_array.sum(0)[np.newaxis, :]
        elif axis == 1:
            ans = self.numpy_array.sum(1)[:, np.newaxis]
        else:
            raise ValueError("axis must be only 0 or 1; instead, got %s\n", axis)

        ans = CUDAMatrix(ans)

        if target is not None:
            target.assign(ans)
        else:
            target = ans
        return target


    def mean(self, axis, target = None):




        if axis == 0:
            ans = self.numpy_array.mean(0)[np.newaxis, :]
        elif axis == 1:
            ans = self.numpy_array.mean(1)[:, np.newaxis]
        else:
            raise ValueError("axis must be only 0 or 1; instead, got %s\n", axis)

        ans = CUDAMatrix(ans)

        if target is not None:
            target.assign(ans)
        else:
            target = ans
        return target





    def assign_sum(self, mat, axis):
        return mat.sum(axis, target = self)

    def assign_mean(self, mat, axis):
        return mat.mean(axis, target = self)



    def add_sums(self, mat, axis, mult = 1.):
        """
        Add a multiple of the sums of the matrix mat along the given dimension
        to self. 
        """



        if self.numpy_array.shape != self.mat.shape:
            raise IncompatibleDimensionsException

        sum = mat.sum(axis)

        sum.numpy_array *= mult

        if axis == 0:
            self.add_row_vec(sum)
        elif axis == 1:
            self.add_col_vec(sum)

        return self


    def less_than(self, val, target = None):
        """
        Perform the operation target = 1. * (self < val), where val can be a matrix or a scalar.
        """


        if target is None:
            target = self

        target.resize(self.shape)

        if isinstance(val, (int, float, np.float128)):
            target.numpy_array[:] = self.numpy_array < val

        else:
            if val.shape != self.shape:
                raise IncompatibleDimensionsException


            target.numpy_array[:] = (self.numpy_array < val.numpy_array).astype(__DTYPE__)

        return target

    def assign_less_than(self, mat, val):
        return mat.less_than(val, self)




    def greater_than(self, val, target = None):
        """
        Perform the operation target = 1. * (self > val), where val can be a matrix or a scalar.
        """


        if target is None:
            target = self

        target.resize(self.shape)

        if isinstance(val, (int, float, np.float128)):
            target.numpy_array[:] = (self.numpy_array > val).astype(__DTYPE__)
        else:
            if val.shape != self.shape:
                raise IncompatibleDimensionsException


            target.numpy_array[:] = (self.numpy_array > val.numpy_array).astype(__DTYPE__)

        return target


    def assign_greater_than(self, mat, val):
        return mat.greater_than(val, self)




    def max(self, axis, target = None, transpose_aux=None):
        """
        Find the maximum value along the given dimension, where 0 represents the
        leading dimension and 1 represents the non-leading dimension. If a target
        is not prvided, a new vector is created for storing the result.
        """



        m, n = self.shape

        if axis == 0:
            if target is None:
                target = empty((1, n))

            target.resize((1, n))

 
            target.numpy_array[:] = self.numpy_array.max(0)

        

        elif axis == 1:
            # IN theory: we are supposed to do this:

#             if not target:
#                 #target = CUDAMatrix(np.empty((m, 1), dtype=np.float32, order = 'F'))
#                 target = empty((m, 1))
#             else:
#                 target.resize((m, 1))
                


#             err_code =  _cudamat.max_by_axis(self.p_mat, target.p_mat, ct.c_int(axis))
#             if err_code:
#                 raise generate_exception(err_code)

            assert transpose_aux != None

            self.transpose(target = transpose_aux)

            target.reshape(target.shape[::-1])

            transpose_aux.max(axis = 0, target = target)

            target.reshape(target.shape[::-1])




        return target

    def assign_max(self, mat, axis, transpose_aux=None):
        return mat.max(axis, target = self, transpose_aux = transpose_aux)

    def total_max(self):
        row_maxes = empty((1, 1)).assign_max(self, axis = 0)
        return row_maxes.reshape((row_maxes.shape[1], row_maxes.shape[0])).max(axis = 0).asarray()[0,0]

    def total_sum(self):
        return self.numpy_array.sum()


    def sign(self, target = None):

        if target is None:
            target = empty(self.shape)

        target.resize(self.shape)

        target.numpy_array[:] = np.sign(self.numpy_array)

        return target


    def assign_sign(self, a):
        return a.sign(target = self)


    def apply_sigmoid(self, target = None):
        """
        Apply the logistic sigmoid to each element of the matrix.
        """

        return sigmoid(self, target)

    def sigmoid(self, target = None):
        """
        Apply the logistic sigmoid to each element of the matrix.
        """

        return sigmoid(self, target)


    def assign_sigmoid(self, t):
        return sigmoid(t, self)


    def log(self, target = None):
        return log(self, target)

    def assign_log(self, t):
        return log(t, self)

    def exp(self, target = None):
        return exp(self, target)

    def assign_exp(self, t):
        return exp(t, self)

    def pow(self, p, target = None):
        return pow(self, p, target)

    def assign_pow(self, mat, p):
        return pow(mat, p, self)


    def sqrt(self, target = None):
        return sqrt(self, target)


    def assign_sqrt(self, mat):
        return sqrt(mat, self)


    def reciprocal(self, target = None):
        """
        Find the reciprocal of each element of the matrix.
        """

        if not target:
            target = self

        target.resize(self.shape)


        target.numpy_array[:] = 1./self.numpy_array[:]

        return target

    def assign_reciprocal(self, mat):
        return mat.reciprocal(target = self)



    def dot(self, mat2, target = None):
        """
        Multiply the matrix by mat2 from the right.
        """

        return dot(self, mat2, target)


    def assign_dot(self, m1, m2):
        m1.dot(m2, target = self)
        return self


    def add_dot(self, m1, m2):
        """
        Add the dot product of m1 and m2 to the matrix.
        """


        m3 = dot(m1, m2)

        if m3.shape != self.shape:
            raise IncompatibleDimensionsException

        self.numpy_array += m3.numpy_array


        return self

    def subtract_dot(self, m1, m2):
        """
        Subtract the dot product of m1 and m2 from the matrix.
        """



        m3 = dot(m1, m2)

        if m3.shape != self.shape:
            raise IncompatibleDimensionsException

        self.numpy_array -= m3.numpy_array


        return self


    def add_mult(self, mat2, alpha = 1.):
        """
        Add multiple of mat2 to the matrix.
        """

        if mat2.shape != self.shape:
            raise IncompatibleDimensionsException

        self.numpy_array += mat2.numpy_array * alpha

        return self
    
    def assign_mult(self, mat2, alpha):
        self.resize(mat2.shape)
        self.assign(0)
        self.add_mult(mat2, alpha)
        return self


    def subtract_mult(self, mat2, alpha = 1.):
        """
        Subtract a multiple of mat2 from the matrix.
        """

        if mat2.shape != self.shape:
            raise IncompatibleDimensionsException

        self.numpy_array -= mat2.numpy_array * alpha

        return self


    def add(self, val, target = None):
        """Add val to self, where val can be a scalar or a CUDAMatrix with the
        same dimensions as self. """

        if not target:
            target = self

        target.resize(self.shape)




        if isinstance(val, CUDAMatrix):
            if target.shape != val.shape:
                raise IncompatibleDimensionsException
            target.numpy_array[:] = self.numpy_array + val.numpy_array

        elif isinstance(val, (int, float, np.float128)):
            target.numpy_array[:] = self.numpy_array + val
        else:
            raise ValueError, "Value must be of type CUDAMatrix, int, or float."



        return target

    def assign_add(self, a, b):
        a.add(b, target = self)
        return self



    def subtract(self, val, target = None):
        """Subtract val from self, where val can be a scalar or a CUDAMatrix with
        the same dimensions as self. """

        if not target:
            target = self

        target.resize(self.shape)



        if isinstance(val, CUDAMatrix):
            if target.shape != val.shape:
                raise IncompatibleDimensionsException
            target.numpy_array[:] = self.numpy_array - val.numpy_array

        elif isinstance(val, (int, float, np.float128)):
            target.numpy_array[:] = self.numpy_array - val
        else:
            raise ValueError, "Value must be of type CUDAMatrix, int, or float."



        return target



    def assign_subtract(self, a, b):
        a.subtract(b, target = self)
        return self




    def divide(self, val, target = None):
        """Divide self by val, where val can be a scalar or a CUDAMatrix with the
        same dimensions as self. """

        if not target:
            target = self

        target.resize(self.shape)


        if isinstance(val, CUDAMatrix):
            if target.shape != val.shape:
                raise IncompatibleDimensionsException
            target.numpy_array[:] = self.numpy_array / val.numpy_array

        elif isinstance(val, (int, float, np.float128)):
            target.numpy_array[:] = self.numpy_array / val
        else:
            raise ValueError, "Value must be of type CUDAMatrix, int, or float."



        return target



    def assign_divide(self, a, b):
        a.divide(b, target = self)
        return self



    def mult(self, val, target = None):
        """Multiply self by val, where val can be a scalar or a CUDAMatrix with
        the same dimensions as self. """

        if not target:
            target = self

        target.resize(self.shape)


        if isinstance(val, CUDAMatrix):
            if target.shape != val.shape:
                raise IncompatibleDimensionsException
            target.numpy_array[:] = self.numpy_array * val.numpy_array

        elif isinstance(val, (int, float, np.float128)):
            target.numpy_array[:] = self.numpy_array * val
        else:
            raise ValueError, "Value must be of type CUDAMatrix, int, or float."



        return target





    def assign_mult(self, a, b):
        a.mult(b, target = self)
        return self




    @deprecated
    def assign_scalar(self, alpha):
        """
        Assign scalar alpha to every element of the matrix.
        """
        self.assign(alpha)
        return self

    @deprecated
    def mult_by_scalar(self, alpha, target = None):
        """
        Multiply the matrix by a scalar.
        """
        return self.mult(alpha, target)




    @deprecated
    def div_by_scalar(self, alpha, target = None):
        """
        Divide the matrix by a scalar.
        """
        
        return self.divide(alpha, target)



    @deprecated
    def add_scalar(self, alpha, target = None):
        """
        Increment the matrix by a scalar.
        """
        return self.add(alpha, target)


    def euclid_norm(self):
        return np.sqrt((self.numpy_array**2).sum())


def empty(shape=None):
    """
    Creates and returns a new CUDAMatrix with the given shape.
    """

    if shape is None:
        shape = (1, 1)

    return CUDAMatrix(np.empty(shape))


def zeros(shape):
    return empty(shape).assign(0)

def randn(a, b):
    ans = empty((a, b)).fill_with_randn()
    return ans



def sum(mat, axis, target = None):
    """
    Sum the matrix along the given dimension, where 0 represents the leading
    dimension and 1 represents the non-leading dimension. If a target is
    not prvided, a new vector is created for storing the result.
    """
    return mat.sum(axis, target)


def dot(m1, m2, target = None):
    """
    Find the dot product between m1 and m2.
    """

    m = m1.shape[0]
    n = m2.shape[1]

    target_shape = (m, n)
    if not target:
        target = empty(target_shape)

    target.resize(target_shape)

    try:
        target.numpy_array[:] = np.dot(m1.numpy_array, m2.numpy_array)
    except ValueError:
        raise IncompatibleDimensionsException

    return target

def vdot(m1, m2):
    assert m1.shape == m2.shape
    return (m1.asarray() * m2.asarray()).sum()



def sigmoid(mat, target = None):
    """
    Apply the logistic sigmoid to each element of the matrix mat.
    """


    if not target:
        target = mat

    target.resize(mat.shape)

    target.numpy_array[:] = 1. / (1 + np.exp(-mat.numpy_array))

    return target


def tanh(mat, target = None):
    """
    Apply the logistic sigmoid to each element of the matrix mat.
    """


    if not target:
        target = mat

    target.resize(mat.shape)

    target.numpy_array[:] = np.tanh(mat.numpy_array)

    return target


def gammaln(mat, target = None):



    if not target:
        target = mat

    target.resize(mat.shape)

    import scipy.special
    target.numpy_array[:] = scipy.special.gammaln(mat.numpy_array)

    return target





def abs(mat, target = None):
    """
    Apply the logistic sigmoid to each element of the matrix mat.
    """


    if not target:
        target = mat

    target.resize(mat.shape)

    target.numpy_array[:] = abs(mat.numpy_array)

    return target




def log_1_plus_exp(mat, target = None):
   """
   Apply log(1+exp(x)) to each element of the matrix mat.
   """
   if not target:
       target = mat
   mask = mat.numpy_array > 0
   target.numpy_array[mask] = mat.numpy_array[mask] + np.log(1+np.exp(-mat.numpy_array[mask]))
   mask = np.logical_not(mask)
   target.numpy_array[mask] = np.log(1+np.exp(mat.numpy_array[mask]))
   return target
log_1_sum_exp = log_1_plus_exp

def log(mat, target = None):
    """
    Find the natural logarithm of each element of the matrix mat.
    """

    if not target:
        target = mat

    target.resize(mat.shape)

    target.numpy_array[:] = np.log(mat.numpy_array)

    return target

def exp(mat, target = None):
    """
    Apply the exponential function to each element of the matrix mat.
    """

    if not target:
        target = mat

    target.resize(mat.shape)

    target.numpy_array[:] = np.exp(mat.numpy_array)

    return target


    if not target:
        target = mat

    target.resize(mat.shape)

    return target


def sqrt(mat, target = None):
    """
    Compute the square root of each element of the matrix mat.
    """

    if not target:
        target = mat

    target.resize(mat.shape)

    target.numpy_array[:] = np.sqrt(mat.numpy_array)

    return target


    if not target:
        target = mat

    target.resize(mat.shape)

    return target

def pow(mat, p, target = None):
    """
    Compute the 'p'th power of each element of the matrix mat.
    """

    if not target:
        target = mat

    target.resize(mat.shape)

    target.numpy_array[:] = mat.numpy_array[:] ** p

    return target

def cuda_sync_threads():
    pass

def reformat(array):
    """
    Returns array as a float32 array in FORTRAN order.
    """
    return np.array(array, dtype=__DTYPE__, order='F')


def cuda_set_some_device():
    return 0

def cuda_set_device(dev_id):
    """
    Selects the CUDA device with the given ID.
    """


    return 0

def cuda_get_free_device():
    """
    Returns the ID of the first free CUDA device.
    """
    return 0



def cublas_init():
    """
    Initialize Cublas.
    """

    return 0

def cublas_shutdown():
    """
    Shut down Cublas.
    """
    return 0


# The following functions are for implementing things like coarse filters and
# models with replicated local filters. At the moment they are quite slow. 

def sum_superpixels(source, target, w, temp = None):
    raise NotImplemented()



def kronecker(mat1, mat2, target = None):
    raise NotIMplemented



def flat_to_tiled(source, target, stride):
    raise NotImplemented()

def tiled_to_flat(source, target, stride, temp = None):
    raise NotImplemented()

def flat_to_tiled3(source, target, stride):
    raise NotImplemented()






def get_item_from_each_row(source, target, inds, num_rows, num_cols):
    if source.numpy_array.shape == (num_cols, num_rows):
        src = source.numpy_array.T
    else:
        src = source.numpy_array.reshape(num_rows, num_cols)
    ix = inds.numpy_array.reshape(num_rows).astype(int)
    t = target.numpy_array.reshape(num_rows)

    for i in range(num_rows):
        t[i] = src[i,ix[i]] 
    return target


def set_item_to_each_row(source, target, inds, num_rows, num_cols):
    if source.numpy_array.shape == (num_cols, num_rows):
        src = source.numpy_array.T
    else:
        src = source.numpy_array.reshape(num_rows, num_cols)

    ix = inds.numpy_array.reshape(num_rows).astype(int)
    t = target.numpy_array.reshape(num_rows)

    for i in range(num_rows):
        src[i,ix[i]] = t[i]
    return source
















def abs(X, aux):
    return aux.assign_mult(X, X).sqrt()

def total_sum(X):
    return X.total_sum()
 

def mean(mat, axis, target = None):

    target = sum(mat, axis, target)



    target.mult_by_scalar(1. / mat.shape[axis])

    return target


def total_mean(mat):
    s = total_sum(mat)
    return s / mat.num_elems








def cumsum(mat, target):
    
    target.resize(mat.shape)

    target.numpy_array[:] = mat.numpy_array.cumsum(1)

    return target




# def multi_transpose(IN, OUT, w, h, batch_size):
#     """
#     the order of w, h seems wrong, but it is consistent with the one on cudamat.py
#     """
#     assert IN.shape == (w*h, batch_size)
#     assert OUT.shape == (w*h, batch_size)


#     from pylab import amap, transpose
#     OUT.numpy_array[:] = amap(transpose,IN.numpy_array.reshape(h, w, batch_size).transpose([2,0,1])).transpose([1,2,0]).reshape(w*h, batch_size)


def multi_transpose(IN, OUT, w, h, batch_size):
    i = IN.numpy_array
    o = OUT.numpy_array
    
#     o = o.reshape(batch_size, w, h) 
#     o[:] = i.reshape(batch_size, h, w).transpose([0,2,1])
#     OUT.numpy_array[:] = o.reshape(*OUT.numpy_array.shape)

    o = o.ravel()
    o[:] = i.reshape(h, w, batch_size).transpose([1,0,2]).ravel()
    OUT.numpy_array[:] = o.reshape(*OUT.numpy_array.shape)


    return OUT

def ind_incr(target, inds, axis):
    

    assert target.shape[1] == inds.shape[0] * inds.shape[1]
    assert inds.shape[1] == 1 or inds.shape[0] == 1

    if axis == 1:
        try:
            for i in inds:
                target.numpy_array[:, i] += 1
        except IndexError:
            raise IncompatibleDimensionsException


        return target

    elif axis == 0:

        try:
            for i in inds:
                target.numpy_array[i, :] += 1
        except IndexError:
            raise IncompatibleDimensionsException


        return target


    else:
        raise Exception ("bad axis.")




## The code below has been lifted from cudamat. It needs to work with numpy.


MAX_ELEMS = 2 ** 16 - 10
class softmax:
    def __init__(self, axis):
        self.axis = axis

        self.transpose_aux = empty()
        self.neg_max = empty()
        self.mat = empty()
        self.exp = empty()
        self.Z = empty()
        self.probs = empty()


        self.transpose_aux_small = empty()
        self.neg_max_small = empty()
        self.mat_small = empty()
        self.exp_small = empty()
        self.Z_small = empty()
        self.probs_small = empty()


    
    def __call__(self, mat, target):


        if mat.shape != target.shape:
            target.resize(mat.shape)

        if self.axis == 1:
            return self.__call_helper_small__(mat, target)




        pos = 0
        step = MAX_ELEMS

        ## width is how many elems we have to work with.
        width = mat.shape[1 - self.axis]

        while pos < width:
            next = min(width, pos + step)

            step_size = next - pos

            if step_size == step:
                self.__call_helper__(mat.slice(pos, next),
                                     target.slice(pos, next))
            else:
                self.__call_helper_small__(mat.slice(pos, next),
                                           target.slice(pos, next))

            pos += step_size

        return target



    def __call_helper__(self, mat, target):






        self.neg_max.\
            assign_max(mat, 
                       axis = self.axis,
                       transpose_aux = self.transpose_aux).\
            mult(-1)

        if self.axis == 0:
            self.mat.assign_add_row_vec(mat, self.neg_max)
        else:
            self.mat.assign_add_col_vec(mat, self.neg_max)

        self.exp.assign_exp(self.mat)

        self.Z.assign_sum(self.exp, self.axis).reciprocal()

        self.probs.assign(self.exp)
        if self.axis == 0:
            self.probs.mult_by_row(self.Z)
        else:
            self.probs.mult_by_col(self.Z)

        target.assign(self.probs)
        



    def __call_helper_small__(self, mat, target):

        self.neg_max_small.\
            assign_max(mat, 
                       axis = self.axis,
                       transpose_aux = self.transpose_aux_small).\
            mult(-1)

        if self.axis == 0:
            self.mat_small.assign_add_row_vec(mat, self.neg_max_small)
        else:
            self.mat_small.assign_add_col_vec(mat, self.neg_max_small)

        self.exp_small.assign_exp(self.mat_small)

        self.Z_small.assign_sum(self.exp_small, self.axis).reciprocal()



        self.probs_small.assign(self.exp_small)
        if self.axis == 0:
            self.probs_small.mult_by_row(self.Z_small)
        else:
            self.probs_small.mult_by_col(self.Z_small)





        target.assign(self.probs_small)

        







    def log_Zs(self, mat, target):

        self.neg_max.\
            assign_max(mat, 
                       axis = self.axis,
                       transpose_aux = self.transpose_aux).\
            mult(-1)

        if self.axis == 0:
            self.mat.assign_add_row_vec(mat, self.neg_max)
        else:
            self.mat.assign_add_col_vec(mat, self.neg_max)

        ## the exps without the max
        self.exp.assign_exp(self.mat)

        ## take the sums of the exps, take the log, and add subtruct the maxes.
        target.assign_sum(self.exp, self.axis).log().add(self.neg_max.mult(-1))

        return target

        






class sample_multinomial:
    def __init__(self, probs, axis):
        raise NotImplementedError("use robust_multinomial instead.")

        self.axis = axis
        self.cumsums = empty()
        self.cumsums_t = empty()
        self.probs_t = empty()
        


        self.cumsums_small = empty()
        self.cumsums_t_small = empty()
        self.probs_t_small = empty()
        




        self.set_probs(probs)
        

        self.samples = empty()
        self.samples_small = empty()


        if axis == 0:

            width = probs.shape[1] 
            std_width = min(width, MAX_ELEMS)



            self.rand_vals = empty((1, std_width))
            self.ones      = empty((probs.shape[0], 1)).assign(1.)



            small_width = max(0, width % MAX_ELEMS)



            self.rand_vals_small = empty((1, small_width))
            self.ones_small      = empty((probs.shape[1], 1)).assign(1.)



        elif axis == 1:


            width = probs.shape[0] 
            std_width = min(width, MAX_ELEMS)



            self.rand_vals = empty((std_width, 1))
            self.ones      = empty((1, probs.shape[1])).assign(1.)



            small_width = max(0, width % MAX_ELEMS)


            self.rand_vals_small = empty((small_width, 1))
            self.ones_small      = empty((1, probs.shape[1])).assign(1.)







        self.rand_mat = empty()
        self.threshs = empty()


        self.rand_mat_small = empty()
        self.threshs_small = empty()




    def set_probs(self, probs):
        if self.axis == 1:
            cumsum(probs, self.cumsums)

        else:
            probs.transpose(target = self.probs_t)
            cumsum(self.probs_t, self.cumsums_t)
            self.cumsums_t.transpose(target = self.cumsums)









    def multi_sample(self, target, k):
        target.resize(self.cumsums.shape)


        for i in range(k):

            self.rand_vals.fill_with_rand()

            if self.axis == 1:
                self.rand_mat.assign_dot(self.rand_vals, self.ones)
            else:
                self.rand_mat.assign_dot(self.ones, self.rand_vals)


            self.threshs.\
                assign_less_than(self.cumsums, self.rand_mat).\
                sum(self.axis, target = self.samples)


            

            ind_incr(target, self.samples, self.axis)

        return target









    def set_probs_helper_small(self, probs):
        self.probs = probs
        if self.axis == 1:
            cumsum(probs, self.cumsums_small)

        else:
            probs.transpose(target = self.probs_t_small)
            cumsum(self.probs_t_small, self.cumsums_t_small)
            self.cumsums_t_small.transpose(target = self.cumsums_small)



    def multi_sample_helper_small(self, target, k):
        target.resize(self.cumsums_small.shape)


        for i in range(k):

            self.rand_vals_small.fill_with_rand()

            if self.axis == 1:
                self.rand_mat_small.assign_dot(self.rand_vals_small, self.ones_small)
            else:
                self.rand_mat_small.assign_dot(self.ones_small, self.rand_vals_small)


            self.threshs_small.\
                assign_less_than(self.cumsums_small, self.rand_mat_small).\
                sum(self.axis, target = self.samples_small)


            

            ind_incr(target, self.samples_small, self.axis)

        return target






    def sample_from_probs(self, probs, target):

        if probs.shape != target.shape:
            target.resize(probs.shape)

        
        ## yes: we make a loop. 

        pos = 0
        step = MAX_ELEMS
        width = probs.shape[1]
        while pos < width:
            next = min(width, pos + step)

            step_size = next - pos

            if step_size == step:
                p = probs.slice(pos, next)
                t = target.slice(pos, next)

                self.set_probs(p)
                self.multi_sample(t, 1)

            else:
                p = probs.slice(pos, next)
                t = target.slice(pos, next)


                self.set_probs_helper_small(probs)
                self.multi_sample_helper_small(t, 1)

            pos += step_size



        return target






class robust_multinomial:
    def __init__(self, shape, axis):
        self.axis = axis
        self.cumsums = empty()
        self.cumsums_t = empty()
        self.probs_t = empty()
        


        self.cumsums_small = empty()
        self.cumsums_t_small = empty()
        self.probs_t_small = empty()
        





        self.samples = empty()
        self.samples_small = empty()


        if axis == 0:

            width = shape[1] 
            std_width = min(width, MAX_ELEMS)



            self.rand_vals = empty((1, std_width))
            self.ones      = empty((shape[0], 1)).assign(1.)



            small_width = max(0, width % MAX_ELEMS)



            self.rand_vals_small = empty((1, small_width))
            self.ones_small      = empty((shape[0], 1)).assign(1.)



        elif axis == 1:


            width = shape[0]
            std_width = min(width, MAX_ELEMS)



            self.rand_vals = empty((std_width, 1))
            self.ones      = empty((1, shape[1])).assign(1.)



            small_width = max(0, width % MAX_ELEMS)


            self.rand_vals_small = empty((small_width, 1))
            self.ones_small      = empty((1, shape[1])).assign(1.)







        self.rand_mat = empty()
        self.threshs = empty()


        self.rand_mat_small = empty()
        self.threshs_small = empty()




    def set_probs(self, probs):
        self.probs = probs
        if self.axis == 1:
            cumsum(probs, self.cumsums)

        else:
            probs.transpose(target = self.probs_t)
            cumsum(self.probs_t, self.cumsums_t)
            self.cumsums_t.transpose(target = self.cumsums)









    def multi_sample(self, target, k):
        target.resize(self.cumsums.shape)


        for i in range(k):

            self.rand_vals.fill_with_rand()

            if self.axis == 1:
                self.rand_mat.assign_dot(self.rand_vals, self.ones)
            else:
                self.rand_mat.assign_dot(self.ones, self.rand_vals)


            self.threshs.\
                assign_less_than(self.cumsums, self.rand_mat).\
                sum(self.axis, target = self.samples)


            

            ind_incr(target, self.samples, self.axis)

        return target









    def set_probs_helper_small(self, probs):
        if self.axis == 1:
            cumsum(probs, self.cumsums_small)

        else:
            probs.transpose(target = self.probs_t_small)
            cumsum(self.probs_t_small, self.cumsums_t_small)
            self.cumsums_t_small.transpose(target = self.cumsums_small)




    def multi_sample_helper_small(self, target, k):
        target.resize(self.cumsums_small.shape)

        for i in range(k):

            self.rand_vals_small.fill_with_rand()

            if self.axis == 1:
                self.rand_mat_small.assign_dot(self.rand_vals_small, self.ones_small)
            else:
                self.rand_mat_small.assign_dot(self.ones_small, self.rand_vals_small)


            self.threshs_small.\
                assign_less_than(self.cumsums_small, self.rand_mat_small).\
                sum(self.axis, target = self.samples_small)


            

            ind_incr(target, self.samples_small, self.axis)

        return target






    def sample_from_probs(self, probs, target):

        if probs.shape != target.shape:
            target.resize(probs.shape)

        
        ## yes: we make a loop. 

        pos = 0
        step = MAX_ELEMS

        width = probs.shape[1 - self.axis]

        while pos < width:
            next = min(width, pos + step)

            step_size = next - pos

            p = probs.slice(pos, next)
            t = target.slice(pos, next)


            if step_size == step:

                self.set_probs(p)
                self.multi_sample(t, 1)

            else:

                self.set_probs_helper_small(p)
                self.multi_sample_helper_small(t, 1)

            pos += step



        return target


#### A few operations for our convolutional friends. Just remember:
#### In cudamat, the data has the form (features, batch_size) as opposed
#### to the numpy order of (batch_size, features) that we are used to. 


def _sqrt(x):
    y = int(x**0.5)
    assert y**2==x
    return y

def _div(a,b):
    assert a%b==0
    return a/b

def true_reshape(numpy_array, shape):
    assert len(shape)==2
    return  numpy_array.reshape(*shape[::-1]).T



def rot180(cu_inputs, cu_targets, color=0):

    """
    npmat has a very deep incompatibility with cudamat, so it's amazing that it 
    works thus far. Basically, reshape has an entirely different semantics.
    
    This reshape is row-column major, the other wise is based on columns.
    
    I know how to fix it, but it would be a big pain. I do so one day. 
    Eventually. Maybe this day will even come. 
    """

    im_pixels, num_imgs = cu_inputs.shape
    im_pixels1, num_imgs1 = cu_targets.shape
    
    assert num_imgs==num_imgs1

    color_mult=[1,3][color]

    img_size=_sqrt(_div(im_pixels,color_mult))

    inputs = cu_inputs.numpy_array
    
    #inputs = true_reshape(inputs, (num_imgs*color_mult, img_size**2))
    targets = np.array([x.reshape(color_mult,img_size,img_size)[:,::-1,::-1].ravel()
                        for x in true_reshape(inputs, (num_imgs, im_pixels))])

    cu_targets.numpy_array[:] = true_reshape(targets, cu_targets.numpy_array.shape)

    return cu_targets


def copy_into_center(cu_images, cu_targets, padding, color=0):
    images_tot_res, num_images = cu_images.shape
    targets_tot_res, num_targets = cu_targets.shape

    assert num_images == num_targets

    color_mult = [1,3][color]
    images_size = _sqrt(_div(images_tot_res,color_mult))
    targets_size = _sqrt(_div(targets_tot_res,color_mult))

    imgs = true_reshape(cu_images.numpy_array, (num_images, color_mult*images_size**2))
    targs = true_reshape(cu_targets.numpy_array, (num_images, color_mult*targets_size**2))
    

    #targs[:, padding:-padding, padding:-padding, :]=imgs
    def pad_local(x):
        ans = np.zeros((color_mult, targets_size, targets_size))
        ans[:,padding:-padding,padding:-padding] = x.reshape((color_mult, images_size, images_size))
        return ans.ravel()

    targs = np.array([pad_local(x)
                      for x in imgs])


    ## that's cool. I think this one is going to work. That is: transpose before reshape. 
    ## or: friends dont let friends reshape before transposing. 
    cu_targets.numpy_array[:] = true_reshape(targs, cu_targets.numpy_array.shape)

    return cu_targets



def add_into_center(cu_images, cu_targets, padding, color=0):
    images_tot_res, num_images = cu_images.shape
    targets_tot_res, num_targets = cu_targets.shape

    assert num_images == num_targets

    color_mult = [1,3][color]
    images_size = _sqrt(_div(images_tot_res,color_mult))
    targets_size = _sqrt(_div(targets_tot_res,color_mult))

    imgs = true_reshape(cu_images.numpy_array, (num_images, color_mult*images_size**2))
    targs = true_reshape(cu_targets.numpy_array, (num_images, color_mult*targets_size**2))
    

    #targs[:, padding:-padding, padding:-padding, :]=imgs
    def pad_local(x,y):
        ans = y.reshape((color_mult, targets_size, targets_size))
        ans[:,padding:-padding,padding:-padding] += x.reshape((color_mult, images_size, images_size))
        return ans.ravel()

    targs = np.array([pad_local(x,y)
                      for (x,y) in zip(imgs,targs)])

    ## that's cool. I think this one is going to work. That is: transpose before reshape. 
    ## or: friends dont let friends reshape before transposing. 
    cu_targets.numpy_array[:] = true_reshape(targs, cu_targets.numpy_array.shape)

    return cu_targets



def copy_out_of_center(cu_images, cu_targets, padding, color=0):
    images_tot_res, num_images = cu_images.shape
    targets_tot_res, num_targets = cu_targets.shape

    assert num_images == num_targets

    color_mult = [1,3][color]
    images_size = _sqrt(_div(images_tot_res,color_mult))
    targets_size = _sqrt(_div(targets_tot_res,color_mult))

    imgs = true_reshape(cu_images.numpy_array, (num_images, color_mult*images_size**2))
    targs = true_reshape(cu_targets.numpy_array, (num_images, color_mult*targets_size**2))
    

    #targs[:, padding:-padding, padding:-padding, :]=imgs
    def pad_local(x):
        ans = np.zeros((color_mult, targets_size, targets_size))
        ans[:] = x.reshape((color_mult, images_size, images_size))[:,padding:-padding,padding:-padding]
        return ans.ravel()

    targs = np.array([pad_local(x)
                      for x in imgs])


    ## that's cool. I think this one is going to work. That is: transpose before reshape. 
    ## or: friends dont let friends reshape before transposing. 
    cu_targets.numpy_array[:] = true_reshape(targs, cu_targets.numpy_array.shape)

    return cu_targets

 
def add_out_of_center(cu_images, cu_targets, padding, color=0):
    images_tot_res, num_images = cu_images.shape
    targets_tot_res, num_targets = cu_targets.shape

    assert num_images == num_targets

    color_mult = [1,3][color]
    images_size = _sqrt(_div(images_tot_res,color_mult))
    targets_size = _sqrt(_div(targets_tot_res,color_mult))

    imgs = true_reshape(cu_images.numpy_array, (num_images, color_mult*images_size**2))
    targs = true_reshape(cu_targets.numpy_array, (num_images, color_mult*targets_size**2))
    

    #targs[:, padding:-padding, padding:-padding, :]=imgs
    def pad_local(x,y):
        ans = np.zeros((color_mult, targets_size, targets_size))
        y = y.reshape((color_mult, targets_size, targets_size))
        ans[:] = y+x.reshape((color_mult, images_size, images_size))[:,padding:-padding,padding:-padding]
        return ans.ravel()

    targs = np.array([pad_local(x,y)
                      for (x,y) in zip(imgs,targs)])


    ## that's cool. I think this one is going to work. That is: transpose before reshape. 
    ## or: friends dont let friends reshape before transposing. 
    cu_targets.numpy_array[:] = true_reshape(targs, cu_targets.numpy_array.shape)

    return cu_targets

 


def sub_sample(cu_images, cu_targets, factor):
    
    images_tot_res, num_images = cu_images.shape
    targets_tot_res, num_targets = cu_targets.shape

    assert num_images == num_targets

    images_size = _sqrt(images_tot_res)
    targets_size = _sqrt(targets_tot_res)

    imgs = true_reshape(cu_images.numpy_array, (num_images, images_size**2))
    targs = true_reshape(cu_targets.numpy_array, (num_images, targets_size**2))
    

    assert targets_size==_div(images_size,factor)

    def downsample(x):
        ans = np.zeros((targets_size, targets_size))
        s = x.reshape((images_size, images_size))
        for ii in range(factor):
            for jj in range(factor):
                ans[:] += s[ii::factor,jj::factor]

        return ans.ravel()/factor**2

    targs = np.array([downsample(x)
                      for x in imgs])

    cu_targets.numpy_array[:] = true_reshape(targs, cu_targets.numpy_array.shape)

    return cu_targets


    
def super_sample(cu_images, cu_targets, factor):
    
    images_tot_res, num_images = cu_images.shape
    targets_tot_res, num_targets = cu_targets.shape

    assert num_images == num_targets

    images_size = _sqrt(images_tot_res)
    targets_size = _sqrt(targets_tot_res)

    imgs = true_reshape(cu_images.numpy_array, (num_images, images_size**2))
    targs = true_reshape(cu_targets.numpy_array, (num_images, targets_size**2))
    

    assert targets_size==images_size*factor

    def supersample(x):
        ans = np.zeros((targets_size, targets_size))
        s = x.reshape((images_size, images_size))
        for ii in range(factor):
            for jj in range(factor):
                ans[ii::factor,jj::factor] = s[:]

        return ans.ravel()

    targs = np.array([supersample(x) # reconnect to my inner nerd.
                      for x in imgs])

    cu_targets.numpy_array[:] = true_reshape(targs, cu_targets.numpy_array.shape)

    return cu_targets







def grid_to_matrix(cu_images, cu_targets, square_size):
    images_tot_res, num_images = cu_images.shape
    targets_tot_res, num_targets = cu_targets.shape


    regions_per_square = _div(num_targets,num_images)
    assert images_tot_res == square_size**2 * regions_per_square
    assert targets_tot_res == square_size**2

    assert images_tot_res==targets_tot_res * regions_per_square

    images_size = _sqrt(images_tot_res)
    targets_size = square_size

    imgs = true_reshape(cu_images.numpy_array, (num_images, images_size**2))
    #targs = true_reshape(cu_targets.numpy_array, (num_images*regions_per_square, targets_size**2))
    

    def concat(lsts):
        ans = []
        for lst in lsts:
            ans.extend(lst)
        return ans

    def gtm(x):
        s = x.reshape((images_size, images_size))
        anses = []
        for i in range(_div(images_size,square_size)):
            for j in range(_div(images_size,square_size)):
                anses.append(s[square_size*i:square_size*(i+1),
                               square_size*j:square_size*(j+1)].T)
        return anses

    targs = np.array(concat([gtm(x) 
                             for x in imgs]))

    cu_targets.numpy_array[:] = true_reshape(targs, cu_targets.numpy_array.shape)




def matrix_to_grid(cu_targets, cu_images, square_size):

    images_tot_res, num_images = cu_images.shape
    targets_tot_res, num_targets = cu_targets.shape

    regions_per_square = _div(num_targets,num_images)
    assert images_tot_res == square_size**2 * regions_per_square
    assert targets_tot_res == square_size**2

    assert images_tot_res==targets_tot_res * regions_per_square

    images_size = _sqrt(images_tot_res)
    targets_size = square_size

    #imgs = true_reshape(cu_images.numpy_array, (num_images, images_size**2))
    targs = true_reshape(cu_targets.numpy_array, (num_targets, targets_size**2))
    
    def concat(lsts):
        ans = []
        for lst in lsts:
            ans.extend(lst)
        return ans

    def partition(lst):
        lst=list(lst)
        ans = []
        s = 0
        while s < len(lst):
            ans.append(lst[s:s+regions_per_square])
            s += regions_per_square
        assert len(ans)==_div(len(lst),regions_per_square)
        return ans

    from pylab import rand
    def mtg(x):
        ans = np.zeros((images_size, images_size)) # that's where the ans lives.
        anses = []
        k = 0
        assert type(x) is list and len(x) % regions_per_square == 0
        for i in range(_div(images_size,square_size)):
            for j in range(_div(images_size,square_size)):
                
                ans[square_size*i:square_size*(i+1),
                    square_size*j:square_size*(j+1)] =  x[k].reshape(square_size,square_size).T
                k+=1

        return ans

    imgs = np.array(concat(map(mtg, partition(targs))))

    cu_images.numpy_array[:] = true_reshape(imgs, cu_images.numpy_array.shape)





def conv(images, filters, targets, num_groups, color, iorder):
    ## I don't understand. I need to work out an example.

    ## this is the conv operations. It's implemented in high-precision stuff.
    ## that's pretty cool. I like it.

    from pylab import reshape

    images_shape = images.shape[::-1]
    filters_shape = filters.shape[::-1]
    targets_shape = targets.shape[::-1]

    from scipy.weave import converters, inline # but that's for later. For now we'll do a simple python loop thingie.
    assert iorder in [0,1]
    if iorder==1: 
        #int colorMult = color ? 3 : 1;
        #Matrix filters(numFiltersPerGroup * numGroups, filterPixels * colorMult);
        #Matrix images(numImgsPerGroup * numGroups, imgPixels * colorMult);
        #Matrix targets(order == GROUP_FILTER_IMAGE ? numFiltersPerGroup * numGroups : numImgsPerGroup * numGroups,
        #               order == GROUP_FILTER_IMAGE ? numImgsPerGroup * numOutputs   : numFiltersPerGroup * numOutputs);

        color_mult = [1,3][color]

        num_filters_per_group__m__num_groups, filter_pixels__m__color_mult = filters_shape
        num_images_per_group__m__num_groups, image_pixels__m__color_mult = images_shape
        num_images_per_group__m__num_groups, num_filters_per_group__m__num_outputs = targets_shape


        num_filters_per_group = _div(num_filters_per_group__m__num_groups, num_groups)
        num_images_per_group = _div(num_images_per_group__m__num_groups, num_groups)
        num_images_per_group = _div(num_images_per_group__m__num_groups, num_groups)

        filter_pixels = _div(filter_pixels__m__color_mult, color_mult)
        image_pixels = _div(image_pixels__m__color_mult, color_mult)
        num_outputs = _div(num_filters_per_group__m__num_outputs, num_filters_per_group)

        outputs_size = output_size = _sqrt(num_outputs)
        images_size = image_size = _sqrt(image_pixels)
        filters_size = filter_size = _sqrt(filter_pixels) # note: image_pixels aren't multiplied by color_mult here.


        # there has gotta be a true reshape someplace. We bother with these retarded true reshapes
        # in the first place to make cudamat.conv happy. Because of the weird way cudamat is 
        # written. Otherwise we'd be living in happy-land. 
        np_images = true_reshape(images.numpy_array, images.shape[::-1])
        np_filters = true_reshape(filters.numpy_array, filters.shape[::-1])
        np_targets = true_reshape(targets.numpy_array, targets.shape[::-1])

        np_images_high = np_images.reshape(num_groups, num_images_per_group, color_mult, image_size, image_size)
        np_filters_high = np_filters.reshape(num_groups, num_filters_per_group, color_mult, filter_size, filter_size)
        np_targets_high = np_targets.reshape(num_groups, num_images_per_group, num_filters_per_group, output_size, output_size)
        np_targets_high*=0

        def tiny_conv(img, filter, x, y):
            # the fact that the results match in even some of the entries
            # suggest that tiny conv is doing the right thing.
            img = img.reshape(images_size, images_size)
            filter = filter.reshape(filter_size, filter_size)
            return (img[x:x+filter_size, y:y+filter_size]*filter).sum()

        for g in range(num_groups):
            for i in range(num_images_per_group):
                for f in range(num_filters_per_group):
                    for y in range(outputs_size):
                        for x in range(outputs_size):
                            for c in range(color_mult):
                                np_targets_high[g,i,f,x,y] += (np_images_high[g,i,c,
                                                                             x:x+filter_size,
                                                                             y:y+filter_size] * 
                                                              np_filters_high[g,f,c,:,:]).sum()



        np_targets_high = np_targets_high.transpose([1,0,2,3,4])

        targets.numpy_array[:] = true_reshape(np_targets_high, targets.numpy_array.shape)

    elif iorder==0:
        #int colorMult = color ? 3 : 1;
        #Matrix filters(numFiltersPerGroup * numGroups, filterPixels * colorMult);
        #Matrix images(numImgsPerGroup * numGroups, imgPixels * colorMult);
        #Matrix targets(order == GROUP_FILTER_IMAGE ? numFiltersPerGroup * numGroups : numImgsPerGroup * numGroups,
        #               order == GROUP_FILTER_IMAGE ? numImgsPerGroup * numOutputs   : numFiltersPerGroup * numOutputs);

        color_mult = [1,3][color]

        num_filters_per_group__m__num_groups, filter_pixels__m__color_mult = filters_shape
        num_images_per_group__m__num_groups, image_pixels__m__color_mult = images_shape
        num_filters_per_group__m__num_groups, num_images_per_group__m__num_outputs = targets_shape


        num_filters_per_group = _div(num_filters_per_group__m__num_groups, num_groups)
        num_images_per_group = _div(num_images_per_group__m__num_groups, num_groups)

        filter_pixels = _div(filter_pixels__m__color_mult, color_mult)
        image_pixels = _div(image_pixels__m__color_mult, color_mult)
        num_outputs = _div(num_images_per_group__m__num_outputs, num_images_per_group)

        outputs_size = output_size = _sqrt(num_outputs)
        images_size = image_size = _sqrt(image_pixels)
        filters_size = filter_size = _sqrt(filter_pixels) # note: image_pixels aren't multiplied by color_mult here.

        np_images = true_reshape(images.numpy_array, images.shape[::-1])
        np_filters = true_reshape(filters.numpy_array, filters.shape[::-1])
        np_targets = true_reshape(targets.numpy_array, targets.shape[::-1])

        np_images_high = np_images.reshape(num_groups, num_images_per_group, color_mult, image_size, image_size)
        np_filters_high = np_filters.reshape(num_groups, num_filters_per_group, color_mult, filter_size, filter_size)
        np_targets_high = np_targets.reshape(num_groups, num_filters_per_group, num_images_per_group, output_size, output_size)
        np_targets_high*=0

        def tiny_conv(img, filter, x, y):
            # the fact that the results match in even some of the entries
            # suggest that tiny conv is doing the right thing.
            img = img.reshape(images_size, images_size)
            filter = filter.reshape(filter_size, filter_size)
            return (img[x:x+filter_size, y:y+filter_size]*filter).sum()



        for g in range(num_groups):
            for i in range(num_images_per_group):
                for f in range(num_filters_per_group):
                    for y in range(outputs_size):
                        for x in range(outputs_size):
                            for c in range(color_mult):
                                np_targets_high[g,f,i,x,y] += (np_images_high[g,i,c,
                                                                             x:x+filter_size,
                                                                             y:y+filter_size] * 
                                                              np_filters_high[g,f,c,:,:]).sum()

        #np_targets_high = np_targets_high.transpose([1,0,2,3,4]): No transpose in this regime. Weird as hell.
        targets.numpy_array[:] = true_reshape(np_targets_high, targets.numpy_array.shape)

    else:
        raise TypeError("iorder should be 0 or 1 only, instead %s" % iorder)





################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################




def conv2(images, filters, targets, filter_size, num_groups, color, iorder):
    ## I don't understand. I need to work out an example.

    ## this is the conv operations. It's implemented in high-precision stuff.
    ## that's pretty cool. I like it.

    from pylab import reshape

    images_shape = images.shape[::-1]
    filters_shape = filters.shape[::-1]
    targets_shape = targets.shape[::-1]

    from scipy.weave import converters, inline # but that's for later. For now we'll do a simple python loop thingie.
    assert iorder in [0,1]
    if iorder==1: 

#    Matrix images(numImgsPerGroup * numGroups, imgPixels * colorMult);
#    Matrix filters(numImgsPerGroup * numGroups, numFiltersPerGroup * filterPixels); // == targets in test_convolve
#    Matrix targets(numImgsPerGroup, numGroups * numFiltersPerGroup * numOutputs * colorMult);

        color_mult = [1,3][color]

        num_images_per_group__m__num_groups, image_pixels__m__color_mult = images_shape
        num_images_per_group__m__num_groups, num_filters_per_group__m__filter_pixels = filters_shape
        num_images_per_group, num_groups__m__num_filters_per_group__m__output_pixels__m__color_mult = targets_shape



        image_pixels = _div(image_pixels__m__color_mult, color_mult)
        filter_pixels = filter_size**2
        num_filters_per_group = _div(num_filters_per_group__m__filter_pixels, filter_pixels)
        output_pixels = _div(_div(_div(num_groups__m__num_filters_per_group__m__output_pixels__m__color_mult,
                                       num_filters_per_group), num_groups), color_mult)

        
        output_size = _sqrt(output_pixels)
        image_size = _sqrt(image_pixels)
        filter_size = _sqrt(filter_pixels)

        np_images = true_reshape(images.numpy_array, images.shape[::-1])
        np_filters = true_reshape(filters.numpy_array, filters.shape[::-1])
        np_targets = true_reshape(targets.numpy_array, targets.shape[::-1])

        np_images_high = np_images.reshape(num_groups, num_images_per_group, color_mult, image_size, image_size)
        np_filters_high = np_filters.reshape(num_groups, num_images_per_group, num_filters_per_group, filter_size, filter_size)
        np_targets_high = np_targets.reshape(num_images_per_group, num_groups, num_filters_per_group, color_mult, output_size, output_size)
        np_targets_high*=0


        np_filters_high = np_filters_high.reshape(num_images_per_group, num_groups, num_filters_per_group, filter_size, filter_size).transpose([1,0,2,3,4])




        def tiny_conv(img, filter, x, y):
            # the fact that the results match in even some of the entries
            # suggest that tiny conv is doing the right thing.
            img = img.reshape(images_size, images_size)
            filter = filter.reshape(filter_size, filter_size)
            return (img[x:x+filter_size, y:y+filter_size]*filter).sum()

        for g in range(num_groups):
            for i in range(num_images_per_group):
                for f in range(num_filters_per_group):
                    for y in range(output_size):
                        for x in range(output_size):
                            for c in range(color_mult):
                                np_targets_high[i,g,f,c,x,y] += (np_images_high[g,i,c,
                                                                                x:x+filter_size,
                                                                                y:y+filter_size] * 
                                                                 np_filters_high[g,i,f,:,:]).sum()


        
        #np_targets_high = np_targets_high.transpose([1,0,2,3,4,5])

        targets.numpy_array[:] = true_reshape(np_targets_high, targets.numpy_array.shape)

    elif iorder==0:

#    if iorder==1: 

#    Matrix images(numImgsPerGroup * numGroups, imgPixels * colorMult);
#    Matrix filters(numImgsPerGroup * numGroups, numFiltersPerGroup * filterPixels); // == targets in test_convolve
#    Matrix targets(numImgsPerGroup, numGroups * numFiltersPerGroup * numOutputs * colorMult);

        color_mult = [1,3][color]

        num_images_per_group__m__num_groups, image_pixels__m__color_mult = images_shape
        num_filters_per_group__m__num_groups, num_images_per_group__m__filter_pixels = filters_shape
        num_images_per_group, num_groups__m__num_filters_per_group__m__output_pixels__m__color_mult = targets_shape



        image_pixels = _div(image_pixels__m__color_mult, color_mult)
        filter_pixels = filter_size**2
        num_filters_per_group = _div(num_filters_per_group__m__num_groups, num_groups)
        output_pixels = _div(_div(_div(num_groups__m__num_filters_per_group__m__output_pixels__m__color_mult,
                                       num_filters_per_group), num_groups), color_mult)

        
        output_size = _sqrt(output_pixels)
        image_size = _sqrt(image_pixels)
        filter_size = _sqrt(filter_pixels)

        np_images = true_reshape(images.numpy_array, images.shape[::-1])
        np_filters = true_reshape(filters.numpy_array, filters.shape[::-1])
        np_targets = true_reshape(targets.numpy_array, targets.shape[::-1])

        np_images_high = np_images.reshape(num_groups, num_images_per_group, color_mult, image_size, image_size)
        np_filters_high = np_filters.reshape(num_groups, num_filters_per_group, num_images_per_group, filter_size, filter_size)
        np_targets_high = np_targets.reshape(num_images_per_group, num_groups, num_filters_per_group, color_mult, output_size, output_size)
        np_targets_high*=0


        #np_filters_high = np_filters_high.reshape(num_images_per_group, num_groups, num_filters_per_group, filter_size, filter_size).transpose([1,0,2,3,4])




        def tiny_conv(img, filter, x, y):
            # the fact that the results match in even some of the entries
            # suggest that tiny conv is doing the right thing.
            img = img.reshape(images_size, images_size)
            filter = filter.reshape(filter_size, filter_size)
            return (img[x:x+filter_size, y:y+filter_size]*filter).sum()

        for g in range(num_groups):
            for i in range(num_images_per_group):
                for f in range(num_filters_per_group):
                    for y in range(output_size):
                        for x in range(output_size):
                            for c in range(color_mult):
                                np_targets_high[i,g,f,c,x,y] += (np_images_high[g,i,c,
                                                                                x:x+filter_size,
                                                                                y:y+filter_size] * 
                                                                 np_filters_high[g,f,i,:,:]).sum()


        
        #np_targets_high = np_targets_high.transpose([1,0,2,3,4,5])

        targets.numpy_array[:] = true_reshape(np_targets_high, targets.numpy_array.shape)


    else:
        raise TypeError("iorder should be 0 or 1 only, instead %s" % iorder)










def conv3(images, filters, targets, num_groups, color, iorder):
    ## I don't understand. I need to work out an example.

    ## this is the conv operations. It's implemented in high-precision stuff.
    ## that's pretty cool. I like it.

    from pylab import reshape

    images_shape = images.shape[::-1]
    filters_shape = filters.shape[::-1]
    targets_shape = targets.shape[::-1]

    from scipy.weave import converters, inline # but that's for later. For now we'll do a simple python loop thingie.
    assert iorder in [0,1]
    if iorder==1: 

#    Matrix images(imgOrder == GROUP_FILTER_IMAGE ? numFiltersPerGroup * numGroups : numImgsPerGroup * numGroups,
 #                 imgOrder == GROUP_FILTER_IMAGE ? numImgsPerGroup * imgPixels    : numFiltersPerGroup * imgPixels);
#    Matrix filters(numFiltersPerGroup * numGroups, filterPixels * colorMult);
#    Matrix targets(numImgsPerGroup * numGroups, numOutputs*colorMult);

        color_mult = [1,3][color]

        num_images_per_group__m__num_groups, num_filters_per_group__m__image_pixels = images_shape
        num_filters_per_group__m__num_groups, filter_pixels__m__color_mult = filters_shape
        num_images_per_group__m__num_groups, num_outputs__m__color_mult = targets_shape

        
        num_images_per_group = _div(num_images_per_group__m__num_groups, num_groups)
        filter_pixels = _div(filter_pixels__m__color_mult, color_mult)
        num_filters_per_group = _div(num_filters_per_group__m__num_groups, num_groups)
        image_pixels = _div(num_filters_per_group__m__image_pixels, num_filters_per_group) 

        filter_size = _sqrt(filter_pixels)
        image_size = _sqrt(image_pixels)

        output_size = image_size - filter_size + 1

        np_images = true_reshape(images.numpy_array, images.shape[::-1])
        np_filters = true_reshape(filters.numpy_array, filters.shape[::-1])
        np_targets = true_reshape(targets.numpy_array, targets.shape[::-1])

        np_images_high = np_images.reshape(num_groups, num_images_per_group, num_filters_per_group, image_size, image_size)
        np_filters_high = np_filters.reshape(num_groups, num_filters_per_group, color_mult, filter_size, filter_size)
        np_targets_high = np_targets.reshape(num_groups, num_images_per_group, color_mult, output_size, output_size)
        np_targets_high*=0

        # for compatibility's sake. Oh, please please make it work. I am so tired of this shit. 

        np_images_high = np_images_high.reshape( num_images_per_group, num_groups, num_filters_per_group, image_size, image_size).transpose([1,0,2,3,4])

        #np_filters_high = np_filters_high.reshape(num_filters_per_group, num_groups, color_mult, filter_size, filter_size).transpose([1,0,2,3,4])

        def tiny_conv(img, filter, x, y):
            # the fact that the results match in even some of the entries
            # suggest that tiny conv is doing the right thing.
            img = img.reshape(images_size, images_size)
            filter = filter.reshape(filter_size, filter_size)
            return (img[x:x+filter_size, y:y+filter_size]*filter).sum()
        def rot180(x):
            return x[::-1,::-1]


        for g in range(num_groups):
            for i in range(num_images_per_group):
                for f in range(num_filters_per_group):
                    for y in range(output_size):
                        for x in range(output_size):
                            for c in range(color_mult):
                                np_targets_high[g,i,c,x,y] += (np_images_high[g,i,f,
                                                                              x:x+filter_size, 
                                                                              y:y+filter_size] *
                                                               rot180(np_filters_high[g,f,c,:,:])).sum()

        targets.numpy_array[:] = true_reshape(np_targets_high, targets.numpy_array.shape)

    elif iorder==0:

#    Matrix images(imgOrder == GROUP_FILTER_IMAGE ? numFiltersPerGroup * numGroups : numImgsPerGroup * numGroups,
#                  imgOrder == GROUP_FILTER_IMAGE ? numImgsPerGroup * imgPixels    : numFiltersPerGroup * imgPixels);
#    Matrix filters(numFiltersPerGroup * numGroups, filterPixels * colorMult);
#    Matrix targets(numImgsPerGroup * numGroups, numOutputs*colorMult);

        color_mult = [1,3][color]

        num_filters_per_group__m__num_groups, num_images_per_group__m__image_pixels = images_shape
        num_filters_per_group__m__num_groups, filter_pixels__m__color_mult = filters_shape
        num_images_per_group__m__num_groups, num_outputs__m__color_mult = targets_shape

        
        num_filters_per_group = _div(num_filters_per_group__m__num_groups, num_groups)
        num_images_per_group = _div(num_images_per_group__m__num_groups, num_groups)
        filter_pixels = _div(filter_pixels__m__color_mult, color_mult)
        num_outputs = _div(num_outputs__m__color_mult, color_mult)
        image_pixels = _div(num_images_per_group__m__image_pixels, num_images_per_group)
        

        filter_size = _sqrt(filter_pixels)
        image_size = _sqrt(image_pixels)

        output_size = image_size - filter_size + 1

        np_images = true_reshape(images.numpy_array, images.shape[::-1])
        np_filters = true_reshape(filters.numpy_array, filters.shape[::-1])
        np_targets = true_reshape(targets.numpy_array, targets.shape[::-1])

        np_images_high = np_images.reshape(num_groups, num_filters_per_group, num_images_per_group, image_size, image_size)
        np_filters_high = np_filters.reshape(num_groups, num_filters_per_group, color_mult, filter_size, filter_size)
        np_targets_high = np_targets.reshape(num_groups, num_images_per_group, color_mult, output_size, output_size)
        np_targets_high*=0

        # for compatibility's sake. Oh, please please make it work. I am so tired of this shit. 

        #np_images_high = np_images_high.reshape( num_images_per_group, num_groups, num_filters_per_group, image_size, image_size).transpose([1,0,2,3,4])

        #np_filters_high = np_filters_high.reshape(num_filters_per_group, num_groups, color_mult, filter_size, filter_size).transpose([1,0,2,3,4])

        def tiny_conv(img, filter, x, y):
            # the fact that the results match in even some of the entries
            # suggest that tiny conv is doing the right thing.
            img = img.reshape(images_size, images_size)
            filter = filter.reshape(filter_size, filter_size)
            return (img[x:x+filter_size, y:y+filter_size]*filter).sum()
        def rot180(x):
            return x[::-1,::-1]


        for g in range(num_groups):
            for i in range(num_images_per_group):
                for f in range(num_filters_per_group):
                    for y in range(output_size):
                        for x in range(output_size):
                            for c in range(color_mult):
                                np_targets_high[g,i,c,x,y] += (np_images_high[g,f,i,
                                                                              x:x+filter_size, 
                                                                              y:y+filter_size] *
                                                               rot180(np_filters_high[g,f,c,:,:])).sum()

        targets.numpy_array[:] = true_reshape(np_targets_high, targets.numpy_array.shape)

    else:
        raise TypeError("iorder should be 0 or 1 only, instead %s" % iorder)




def expand(labels, n_labels):
    a=np.zeros((len(labels), n_labels))
    for i in range(len(labels)):
        a[i,labels[i]]=1
    return a


def max_row_argmax(inputs, max_targets, argmax_targets):
    assert max_targets.shape[0]==1
    assert inputs.shape[1]==max_targets.shape[1]
    assert argmax_targets.shape == inputs.shape

    x = inputs.numpy_array.T
    np_maxes = x.max(1)
    np_argmaxes = expand(x.argmax(1), x.shape[1])
    max_targets.numpy_array[:] = np_maxes.T
    argmax_targets.numpy_array[:] = np_argmaxes.T

    return max_targets, argmax_targets
