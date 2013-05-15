def load_default(path, closure):
    from pylab import load, save
    try:
        return load(path)
    except IOError: 
        obj = closure()
        save(obj, path)
        return obj
