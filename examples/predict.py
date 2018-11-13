"""
Predict the class of the fundus photographs.

Usage:
    predict [--verbose] <command> [<args>...]

The most commonly used EyesNet commands are:
    field2
    left_right

Examples:
    cat <file> | xargs -d '\n' python examples/predict.py field2 -d deep-learning-models/field2_model.h5 --output <output>
"""
from subprocess import call

from examples.utils import parse_args

if __name__ == '__main__':
    args = parse_args(__doc__, version='EyesNet version 1', options_first=True)
    argv = [args['<command>']] + args['<args>']

    if args['<command>'] == 'field2':
        exit(call(['python', 'examples/predict_field2.py'] + argv))
    if args['<command>'] == 'left_right':
        exit(call(['python', 'examples/predict_left_right.py'] + argv))
    else:
        exit("%r is not a predict.py command. See 'predict help'." % args['<command>'])
