import gnumpy as g

def permute_string(string_name, permute_by, string):
    parts = string.split(permute_by)
    from numpy.random import permutation
    from opt.utils.persistence import load_default
    perm = load_default('opt/d/lang/orig_perm_of%s' % string_name,
                        lambda: permutation(len(parts)))
    new_parts = []
    for p in perm:
        new_parts.append(parts[p])
    return permute_by.join(new_parts)




class ContiguousText(object):
    def __init__(self, name, string, T, batch_size, train_prob=.8, T_warmup=10, T_frac=0.9,
                 explicit_train_string=None, explicit_test_string=None):
        self.test_cost_correction_factor = 1.

        self.name = name

        self.string = string
        self.train_prob = train_prob

        # the train/test split is really, really simple. 

        if string is not None:

            assert explicit_train_string is None and explicit_test_string is None
            cutoff = int(len(string) * train_prob)
            self.train_string = string[:cutoff]
            self.test_string = string[cutoff:] 
        else:
            assert train_prob is None and string is None
            self.train_string = explicit_train_string
            self.test_string = explicit_test_string


        self.chars = chars = \
  'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-+.,:;?/\\!@#$%&*()"\'\n '
        self.chars_dict = dict((c,i) for (i,c) in enumerate(chars))
        self.chars_inv_dict = dict([(i,c) for (i,c) in enumerate(chars)] + [(len(chars), '^')])

        self.batch_size = batch_size
        self.T = T
        assert T >= 2*T_warmup
        self.true_T = min(int(T*T_frac), T-T_warmup) # whichever eats up more: either T_warmup, or T*T_frac
        self.T_warmup = T_warmup

        self.o = self.O = self.v = self.V = len(self.chars) + 1

        from opt.utils.nonlin import Softmax;         self.out_nonlin = Softmax

        self.batch_fn = self

        def calc_num_batches(str):
            from numpy import ceil, floor
            l = float(len(str))
            # that's the true batch size. Awesome. 
            bs = float(batch_size * self.true_T)
            return int(ceil(l/bs))

        self.train_batches = [x for x in 
                              range(calc_num_batches(self.train_string))]

        self.test_batches = [-x-1 for x in 
                              range(calc_num_batches(self.test_string))]

        self.train_mem = dict()


    def print_batch(self, X, b):
        from pylab import find

        if len(X[0][0])==1:
            M = X
            ans = ['01'[int(m[b].asarray())] for m in M]
        else:
            inds = [int(find(x[b].asarray())) for x in X
                    if len(find(x[b].asarray()))>0]
            ans = [self.chars_inv_dict[i] for i in inds]

        return ''.join(ans).replace('\n', ' ').replace('\t', ' ')

    def from_string_to_id(self, string):
        ans = [self.chars_dict.get(ch, len(self.chars)) for ch in string]
        return ans

    def make_V_O_M_seed(self):
        from numpy.random import randint

        ans = []
        T = self.T
        for i in range(self.batch_size): # random substrings
            b = randint(len(self.train_string) - T)
            v = self.train_string[b: b+T+1]
            ans.append(self.from_string_to_id(v))
        return (ans, None)

    def make_V_O_M_test_seed(self, batch_id):
        assert 0 <= batch_id < len(self.test_batches)
        true_batch_size = self.true_T * self.batch_size

        start = batch_id * true_batch_size
        end = min((batch_id + 1) * true_batch_size, 
                  len(self.test_string))
        
        from numpy import zeros
        M = zeros((self.T, self.batch_size, 1)) 
        ans = []
        skip = self.true_T
        for ii in range(self.batch_size):
            a = start + (ii+0)* skip # note that this is prediction range.
            b = start + (ii+1)* skip # so we'll be needing the context before a.

            aa = b - self.T - 1 # this is a start index that includes the context.
            
            # if the context is entirely in the test set then it's easy:
            if aa >= 0:
                # we grab the contexted string
                v = self.test_string[aa:b]
                ans.append(self.from_string_to_id(v))


                if len(v)==self.T+1:
                    # if we're somewhere in the middle (so v is of full length), then M is easy to deal with.
                    M[-self.true_T:, ii] = 1 
                else:
                    
                    # but if we're closer to the end, then we shouldn't have 1's at the M all the way:
                    M[-self.true_T:, ii] = 1
                    # but not next to the most very end, as follows: 
                    extra_length = self.T + 1 - len(v)  # say we're 1 too long
                    M[-extra_length:, ii] = 0   ### then we'll remove the last extra guy.

            else:
                assert a == 0
                # if aa is negative then we're dealing with the very beginning of the 
                # test sequence. That's fine. Then we'll get a slightly shorter string:
                v = self.test_string[0:b]
                ans.append(self.from_string_to_id(v))
                M[:len(v)-1, ii] = 1  # note that we need to predict all the chars, 
                                    # including those occurring right next to the start. 

        return (ans, M)

    def make_V_O_M_expand(self, (V, M)):
        T = self.T

        from pylab import expand

        batch_size = len(V)
        assert len(V) <= self.batch_size 

        from numpy import zeros
        core_V = zeros((T, batch_size, self.V))
        core_O = zeros((T, batch_size, self.O))
        core_M = zeros((T, batch_size, 1))
        core_V_small = zeros((T+1, batch_size, self.V))

        #for b in range(len(V)):
        #    core_V_small[:,b,:] = expand(V[b], self.V)
        for b in range(len(V)):
            t = len(V[b])
            core_V_small[:t, b, :] = expand(V[b], self.V)

            #if t != self.T+1: # that's good. Now let's proceed.
            #    assert M is not None and (M[-1][0]==0).all()

        core_V[:] = core_V_small[:-1]
        core_O[:] = core_V_small[1:]

        if M is None:
            core_M[-self.true_T:, :, :] = 1
        else:
            # the point is, M is known during training. But during test time,
            # right at the beginning, we want M to cover the start as well. 
            # Alright, cool! Let's proceed. 
            core_M[:] = M 
        return core_V, core_O, core_M

    def __call__(self, x):
        if x < 0:
            # test
            batch_id = -x-1
            seed = self.make_V_O_M_test_seed(batch_id)
        else:
            # train
            batch_id = x
            if batch_id in self.train_mem: 
                seed = self.train_mem[batch_id] # get the seed
            else:
                seed = self.make_V_O_M_seed() # or get a new one
                self.train_mem[batch_id] = seed
    
        V_np, O_np, M_np = self.make_V_O_M_expand(seed)

        Vs = map(g.garray, V_np)
        Os = map(g.garray, O_np)
        Ms = map(g.garray, M_np)
        return Vs, Os, Ms

    def sig(self, batch):
        """
Return "an id" for each batch. Used to ensure that different minibatches are 
actually different. 
        """
        ans = 0
        for t in [-1,-2,-3]:
            z = batch[0][t]
            ans += (z[:-1]*z[1:]).sum() 
        return ans

    def forget(self):
        self.train_mem.clear()

    def size(self, b):
        V,O,M = b
        assert len(V)==len(O)==len(M)

        # allow for fractional batch sizes, in case our test set doesn't quite have everything. Awesome. 
        return sum([m.sum() for m in M]) / float(self.true_T)

    def __repr__(self):
        ans = '\n'.join(['A contiguous text dataset of:', 
                         self.name,
                         'batch_size = %s' % self.batch_size,
                         'T = %s' % self.T,
                         'T_warmup = %s' % self.T_warmup,
                         'train_prob = %s' % self.train_prob,
                         'train len = %s' % len(self.train_string),
                         'test len = %s' % len(self.test_string)])
        return ans
                    

    def get_train_file(self):
        def convert(string):
            ans = []
            for s in string:
                if s in self.chars_dict:
                    ans.append(s)
                else:
                    ans.append('^')
            return ''.join(ans)

        print 'writing the string to a tmp file.'
        path = '/ais/gobi2/ilya/tmp/train_text.txt'
        f = file(path, 'w')
        f.write(convert(self.train_string))
        f.close()



    def obtain_memoizer_log_prob(self, remove_tmp_files=True, alpha=5, cheap=False,
                                 train_loss=False, space_to_x=False):
        def convert(string):
            ans = []
            for s in string:
                if s in self.chars_dict:
                    if space_to_x and s==' ':
                        s = 'x'
                    ans.append(s)
                else:
                    ans.append('^')
            return ''.join(ans)

        print 'writing the string to a tmp file.'
        path = '/ais/gobi2/ilya/tmp/train_text'
        f = file(path, 'w')
        f.write(convert(self.train_string))
        f.close()

        path = '/ais/gobi2/ilya/tmp/test_text'
        f = file(path, 'w')
        if train_loss is True: # if we'd like to compute the train loss, that's what
            # we'd do.
            f.write(convert(self.train_string))
        else:
            f.write(convert(self.test_string))
        f.close()

        import os
        pwd = os.getcwd()

        print 'going to the local directory. will spend some time in there.'
        os.chdir('/ais/gobi2/ilya/tmp/')

        if cheap:
            # ./score_file --test-file test_text.txt --mode 2 --restaurant 3 --alpha 5 train_text.txt
            os.spawnl(os.P_WAIT, './score_file', 'score_file', 'train_text', '--test-file', 'test_text', 
                      '--num-types', str(len(self.chars_dict)+1), '--alpha', str(alpha), '--mode', '2', '--restaurant', '3')
        else:
            os.spawnl(os.P_WAIT, './score_file', 'score_file', 'train_text', '--test-file', 'test_text', 
                      '--num-types', str(len(self.chars_dict)+1), '--alpha', str(alpha))
        os.chdir(pwd)
        print 'done. back to the original directory. Please read off the answer yourself'
        print '(in the loss 2 lines above).'
        

        if remove_tmp_files:
            os.remove('/ais/gobi2/ilya/tmp/train_text')
            os.remove('/ais/gobi2/ilya/tmp/test_text')
        else:
            print 'Not removing the temporary files:'
            print '/ais/gobi2/ilya/tmp/train_text'
            print '/ais/gobi2/ilya/tmp/test_text'
            print '\nRemove them yourself.'



    def sample_from_ppmpaq(self, spec):
        """
what's the strategy? 
we'll create the following file:
(self.string, spec[0][0], '~'*spec[0][1], spec[1][0], '~'*spec[1][1],...)
"""
        parts = [self.train_string] # we dont' want to sample thing
        for (prefix, num) in spec:
            parts.append(prefix)
            parts.append('~'*num) # '~' becasue the hacked paq knows to sample at '~'.s

        data = ''.join(parts)
        fname = self.name + '.for_paq_sampling'

        sampling_cutoff = str(len(self.train_string))

        # 
        import os
        pwd = os.getcwd()
        os.chdir('/u/ilya/py/p11/ppmpaq/')

        # write the file.
        f=file(fname, 'w')
        f.write(data)
        f.close()

        target_file_name = self.name + '.samples'

        try:
            os.remove('/u/ilya/py/p11/ppmpaq/%s.paq' % fname) # remove the paq
            print 'successfully removed %s.paq' % fname
        except OSError:
            print '%s.paq wasn\'t around' % fname

        os.spawnl(os.P_WAIT, './a.modified_sample.out', '', '-8', fname+'.paq', fname,
                  '-sampling_cutoff', sampling_cutoff, 
                  '-target_file_name', 
                  target_file_name)

        sample  = file(target_file_name).read(-1)
        # return to the original directory.
        os.chdir(pwd)

        # finish off by collecting the samples.
        ptr = 0
        the_samples = []
        for (prefix, num) in spec:
            assert sample[ptr:ptr+len(prefix)]==prefix
            ptr += len(prefix)
            the_samples.append((prefix, sample[ptr:ptr+num]))
            ptr += num
        return the_samples



    def obtain_ppmpaq_log_prob(self, space_to_tilde=False):
        def convert(string):
            ans = []
            for s in string:
                if s in self.chars_dict:
                    if space_to_tilde and s==' ':
                        s='~'
                    ans.append(s)
                else:
                    ans.append('^')
            return ''.join(ans)


        run_name = self.string[:12].replace('\n', '').replace(' ','').replace('\t', '')

        tmp_input = 'tmp_input_%s.input' % run_name
        log_name = 'log_prob_%s_.txt' % run_name



        print 'writing the string to a tmp file.'
        path = '/ais/gobi2/ilya/tmp/'  + tmp_input
        f = file(path, 'w')
        f.write(convert(self.string))
        f.close()

        print 'writing the training set size.'
        #f = file('/ais/gobi2/ilya/tmp/log_prob_byte_thresh_file.txt', 'w')
        len_train_string = str(len(self.train_string))
        #f.write()
        #f.close()


        import os
        pwd = os.getcwd()

        print 'going to the local directory. will spend some time in there.'
        os.chdir('/ais/gobi2/ilya/tmp/')
        try:
            os.remove('/ais/gobi2/ilya/tmp/%s.paq' % tmp_input) # remove the paq
            print 'successfully removed %s.paq' % tmp_input
        except OSError:
            print '%s.paq wasn\'t around' % tmp_input

        os.spawnl(os.P_WAIT, './a.out.logprob', '', '-8', tmp_input+'.paq', tmp_input,
                  '-test_set_cutoff', len_train_string, '-run_name', log_name)
        os.chdir(pwd)
        print 'done. back to the original directory.'
        

        os.remove('/ais/gobi2/ilya/tmp/%s.paq' % tmp_input)
        ans = file('/ais/gobi2/ilya/tmp/%s' % log_name).readlines()
        self._ans = ans
        from numpy import log
        return float(ans[0])/log(2)









### Finally, I've found a good use of inheritance. 
class ShiftCharContiguousText(ContiguousText):

    def __init__(self, name, string, T, batch_size, train_prob=.8, T_warmup=10):
        self.name = 'ShiftChar: ' + name

        SHIFT = '^'
        UNK = '~'

        def augment(string):
            string = string.replace('^', '~') # as '^' is Shift, and we don't want accidental shifts.
            import re
            ans = re.sub(r'([A-Z])', r'^\1', string).lower()
            return ans

        self.train_prob = train_prob

        self.orig_string = string
        self.string = augment(string)

        cutoff = int(len(string) * train_prob)
        self.train_string = augment(string[:cutoff])
        self.test_string = augment(string[cutoff:])

        len_orig_len_string = float(len(string) - cutoff)

        ## the real problem is that, since we have more characters, the cost per 
        ## character will grow. Therefore we also need a test correction factor that 
        ## will take that into account. 
        self.test_cost_correction_factor = len(self.test_string) / len_orig_len_string


        self.chars = chars = \
  'abcdefghijklmnopqrstuvwxyz0123456789_-+.,:;?/\\!@#$%&*()"\'\n ^' # ~ is the unknown char. cool.

        self.chars_dict = dict((c,i) for (i,c) in enumerate(chars))
        self.chars_inv_dict = dict([(i,c) for (i,c) in enumerate(chars)] + [(len(chars), UNK)])



        self.batch_size = batch_size
        self.T = T
        assert T >= 2*T_warmup
        self.true_T = min(int(T*0.9), T-T_warmup)
        self.T_warmup = T_warmup

        self.o = self.O = self.v = self.V = len(self.chars) + 1

        from opt.utils.nonlin import Softmax;         self.out_nonlin = Softmax

        self.batch_fn = self

        def calc_num_batches(str):
            from numpy import ceil, floor
            l = float(len(str))
            bs = float(batch_size * self.true_T)
            return int(ceil(l/bs))

        self.train_batches = [x for x in 
                              range(calc_num_batches(self.train_string))]

        self.test_batches = [-x-1 for x in 
                              range(calc_num_batches(self.test_string))]

        self.train_mem = dict()



 





class ShiftFeatureContiguousText(ContiguousText):
    def __init__(self, name, string, T, batch_size, train_prob=.8, T_warmup=10):
        self.test_cost_correction_factor = 1.

        self.name = name

        self.string = string
        self.train_prob = train_prob

        cutoff = int(len(string) * train_prob)
        self.train_string = string[:cutoff]
        self.test_string = string[cutoff:] 

        
        self.chars = chars = \
  'abcdefghijklmnopqrstuvwxyz0123456789_-+.,:;?/\\!@#$%&*()"\'\n '
        self.chars_dict = dict((c,i) for (i,c) in enumerate(chars))
        self.chars_inv_dict = dict([(i,c) for (i,c) in enumerate(chars)] + [(len(chars), '~')])

        self.batch_size = batch_size
        self.T = T
        assert T >= 2*T_warmup
        self.true_T = min(int(T*0.9), T-T_warmup)
        self.T_warmup = T_warmup

        # the only extra 1 feature, the uppercase.
        self.o = self.O = self.v = self.V = len(self.chars) + 1 + 1

        from opt.utils.nonlin import Sigmoid, Softmax, Join
        v = self.v
        self.out_nonlin = Join(Softmax, (0, v-1), 
                               Sigmoid, (v-1, v))


        self.batch_fn = self

        def calc_num_batches(str):
            from numpy import ceil, floor
            l = float(len(str))
            bs = float(batch_size * self.true_T)
            return int(ceil(l/bs))

        self.train_batches = [x for x in 
                              range(calc_num_batches(self.train_string))]

        self.test_batches = [-x-1 for x in 
                              range(calc_num_batches(self.test_string))]

        self.train_mem = dict()


    def print_batch(self, X, b):
        from pylab import find
        inds = [int(find(x[b][:-1].asarray())) for x in X]
        uppers = [float(x[b][-1].asarray()) for x in X]

        def up(x,u):
            if u: return x.upper()
            else: return x
        ans = [up(self.chars_inv_dict[i], i_up) 
               for (i, i_up) in zip(inds, uppers)]
        
        return ''.join(ans)


    def from_string_to_id(self, string):
        # get the index of each lowercase string
        ans_letter = [self.chars_dict.get(ch, len(self.chars)) for ch in string.lower()]
        ans_shift = [ch.isupper() for ch in string]
        return (ans_letter, ans_shift)

    def make_V_O_M_expand(self, V):
        T = self.T

        from pylab import expand

        batch_size = len(V)
        assert len(V) <= self.batch_size

        from numpy import zeros, ones
        core_V = zeros((T, batch_size, self.V))
        core_O = zeros((T, batch_size, self.O))
        core_M = ones((T, batch_size, 1))
        core_V_small = zeros((T+1, batch_size, self.V))

        for b in range(len(V)):
            #core_V_small[:,b,:] = expand(V[b], self.V)
            # that's the part that must go and become different: 
            core_V_small[:,b,:-1] = expand(V[b][0], self.v-1)
            core_V_small[:,b,-1] = V[b][1]

        core_V[:] = core_V_small[:-1]
        core_O[:] = core_V_small[1:]

        core_M[:-self.true_T, :, :] = 0
        return core_V, core_O, core_M



    def __repr__(self):
        ans = '\n'.join(['A ShiftFeatureContigous text dataset of:', 
                         self.name,
                         'batch_size = %s' % self.batch_size,
                         'T = %s' % self.T,
                         'T_warmup = %s' % self.T_warmup,
                         'train_prob = %s' % self.train_prob,
                         'train len = %s' % len(self.train_string),
                         'test len = %s' % len(self.test_string)])
        return ans
                    

    def obtain_ppmpaq_log_prob(self):
        raise Exception ("ShiftFeatureContiguousText does not run ppmpaq. "
                         "To get ppmpaq, create a standard ContiguousText "
                         "object and call it from there.")


