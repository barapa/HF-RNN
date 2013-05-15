class _PathMaker:

    def __init__(self):
        self.path_ = '.'
        self.name = None

    def set_path_and_name(self, path, name):
        import os

        path.rstrip('/')


        path_parts = path.split('/')
        cum_path = './'

        for path_part in path_parts:
            cum_path += path_part + '/'
            if os.path.exists(cum_path):
                pass
            else:
                os.mkdir(cum_path)

        self.path_ = path + '/'

        self.name = name

    @property
    def path(self):
        if self.name is None:
            return None

        else:
            return self.path_ + self.name 



class PrintfMaker:
    def __init__(self):
        self.path_maker = _PathMaker()
        self.reset_filename()

    def set_path_and_name(self, path, name):
        self.path_maker.set_path_and_name(path, name)
        self.reset_filename()

    def reset_filename(self):
        file_name = self.path_maker.path
        if file_name is None:
            import sys
            self.file = sys.stdout
        else:
            self.file = file(file_name, 'a')

    def __call__(self, text):
        self.file.write(text)
        self.file.flush() 


def make_cg_printf(path):
    printf = PrintfMaker()
    printf.set_path_and_name(path, 'results')

    printf_cg = PrintfMaker()
    printf_cg.set_path_and_name(path, 'detailed_results')
    return printf, printf_cg 
