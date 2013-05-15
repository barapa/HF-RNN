import cPickle    

def save(p, filename):
    f=file(filename,'wb')
    cPickle.dump(p,f,cPickle.HIGHEST_PROTOCOL) 
    f.close()

def load(filename):
    f=file(filename,'rb')
    y=cPickle.load(f)
    f.close()
    return y
