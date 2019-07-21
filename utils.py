import argparse
import operator

def float_range_argparse_checker(min_val=0., max_val=1., exc_min=False, exc_max=False):
    def checker(val_str):
        val = float(val_str)
        min_op, min_op_str = (operator.gt, '>') if exc_min else (operator.ge, '>=')
        max_op, max_op_str = (operator.lt, '<') if exc_max else (operator.le, '<=')
        if min_op(val, min_val) and max_op(val, max_val):
            return val
        raise argparse.ArgumentTypeError(
            'Value must be {} {} and {} {} (received {})'.format(min_op_str, min_val, max_op_str, max_val, val))
    if min_val >= max_val:
        raise ValueError('min_val must be less than max_val')
    return checker