def show_hids(r, st, b, batch_id=-1, Tr=True, 
              show_min=None, show_max=None, nonlin=None, ret=False):
    W = r.t.W.unpack(r.t.X)
    X, Y, M = r.t.batch_fn(batch_id)

    state = W.forward_pass(X)

    if isinstance(state, (list, tuple)):
        state = dict(enumerate(state))
    assert isinstance(state, dict)

    state['Y'] = Y
    state['O'] = Y
    state['M'] = M



    import numpy as np
    if nonlin is not None:
        from pylab import amap
        hid = [nonlin(h) for h in state[st] 
               if h is not None]
    else:
        hid = [h for h in state[st] 
               if h is not None]


    hid = np.array([x[b].asarray() for x in hid])






    from pylab import show
    if Tr: hid=hid.T
    if ret:
        return hid
    else:
        show(hid, show_min, show_max)
