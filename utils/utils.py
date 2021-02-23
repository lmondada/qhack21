import pandas as pd
import numpy as np

from functools import wraps

class print_progress:
    """
    Small utility to print progress in the form of "TEXT... ok"

    Use it as follows:
     > with print_progress("Downloading"):
     >     # download
    """
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        print(self.name, end='... ', flush=True)
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            print('ok')
        else:
            print(f'ERROR: {exc_type}')

class InvalidIndex(Exception): pass

def support_pandas(f):
    """ Decorator to transform pandas input into numpy form

    Function decorator to convert pandas Series n-dimnesional inputs into n+1 dimensional numpy arrays.
    This way, we can always use numpy internally, whilst having the freedom to use numpy or pandas
    on our other code.

    This is useful because pandas data types are pretty, but computations are easier
    (and faster) to execute directly on numpy data types.
    """
    to_numpy = lambda x : np.stack(x.values)
    to_pandas = lambda x, ind: pd.Series(list(x), index=ind)

    ind = None
    n_data = None
    wrap_output = False

    def convert(arg):
        nonlocal ind, n_data, wrap_output
        if isinstance(arg, pd.Series):
            if ind is not None and not ind.equals(arg.index):
                raise InvalidIndex("Different pandas arguments must have same index")
            ind = arg.index
            wrap_output = True
            arg = to_numpy(arg)
        elif isinstance(arg, np.ndarray):
            if n_data is not None and n_data != len(arg):
                raise InvalidIndex("Different numpy arguments must have same length")
            n_data = arg.shape[0]
        if n_data is not None and ind is not None and n_data != len(ind):
            raise InvalidIndex("Nunpy arrays and pandas Series must have same number of rows")
        return arg

    @wraps(f)
    def inner(*args, **kwargs):
        converted_args = [convert(arg) for arg in args]
        converted_kwargs = {k: convert(kwarg) for k, kwarg in kwargs.items()}

        ret = f(*converted_args, **converted_kwargs)

        return to_pandas(ret, ind) if wrap_output else ret
    return inner