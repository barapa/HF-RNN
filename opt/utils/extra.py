
def random_subset_2(list, num):
    if num>=len(list):
        return list
    else:
        return random_subset(list, num)

def random_subset(list, num):
    from pylab import randint

    assert num <= len(list)

    remaining_inds = range(len(list))
    inds = []
    for n in range(num):
        i = randint(len(remaining_inds))
        actual_ind = remaining_inds[i]
        del remaining_inds[i]

        inds.append(actual_ind)

    return [list[i] for i in inds]


def random_pick(lst):
    import numpy as np
    return lst[np.random.randint(len(lst))]

def sparsify_strict(a, num_in, sb, sc):
    assert type(num_in)==int
    A = a.asarray() * sb 
    from numpy.random import rand, permutation
    

    for i in range(A.shape[1]):
        perm = permutation(A.shape[0])
        BIG = perm[:num_in]
        SMALL = perm[num_in:]
        A[SMALL, i] *= sc / sb

    a[:] = A

def unpack(arrays, X):
    total_size = sum(a.size for a in arrays)
    assert total_size == X.size

    ans_arrays = []
    cur = 0
    for a in arrays:
        ans_arrays.append(
            X[cur:cur + a.size].reshape(a.shape))

        cur += a.size
    return ans_arrays



def unpack_sizes(sizes, X):
    from pylab import prod
    total_size = sum(sizes)
    assert total_size == X.size

    ans_arrays = []
    cur = 0
    for size in sizes:
        ans_arrays.append(
            X[cur:cur + size])

        cur += size
    return ans_arrays




def rand_pick_list(list, prob):
    from numpy.random import rand
    ans = []
    for l in list:
        if rand() < prob:
            ans.append(l)
    return ans



