
data = file('data/texts/moon.txt').read(-1)


from opt.d.lang import ContiguousText
data_object = ContiguousText(name = 'test',
                   
                   string = None,
                   train_prob = None,

                   explicit_train_string = data[:int(1e5)],
                   explicit_test_string = data[:int(1e3)],
               
                   T = 51,
                   batch_size = 7)


import gnumpy as g



def test_cover():
    """
Goal: verify that each character of the test set is covered
precisely once as we iterate through the test batches. Each character
except for the first must appear precisely once in the v, o, m batch thing.
"""

    true_batch_size = data_object.true_T * data_object.batch_size

    string = data_object.test_string

    skip = data_object.true_T
    cover = g.zeros(len(data_object.test_string)).asarray().astype(int)




    for batch_id in data_object.test_batches:
        v, o, m = batch = data_object(batch_id)

        start = (-1-batch_id) * true_batch_size

        for b in range(data_object.batch_size):

            from pylab import find

            def disp(vs):
                return ''.join([(data_object.chars+'^')[int(find(vss[b].asarray()))]
                                for vss in vs
                                if vss[b].sum()!=0])


            for t in range(data_object.T):
                    
                if b==0 and batch_id==-1:
                    ind=t+1
                else:
                    a = start + (b+1)*skip  - data_object.T - 1
                    ind = a+t+1

                
                if ind<0:
                    continue
                try:
                    # this one can give a type error
                    v_ch = (data_object.chars+'^')[int(find(o[t][b].asarray()))]

                    # and this one can give an index error
                    d_ch = data_object.test_string[ind]

                    if d_ch not in data_object.chars:
                        d_ch = '^'

                except (TypeError, IndexError):
                    assert (m[t][b] == 0).all()
                else:
                    if m[t][b]==1:
                        assert v_ch == d_ch
                        cover[ind] = 1

                

    assert (cover[1:]==1).all()

