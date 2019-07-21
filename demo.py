import argparse
import parser
import CompFramework.models as models
import tensorflow as tf

from CompFramework.models.tfnets.resnet_cifar10 import ModelHelper

def main():
    args = parser.get_parser().parse_args()
    if args.epochs is None:
        args.epochs = 90
        args.epochs = 90

    start_epoch = 0
    ending_epoch = args.epochs

    #Todo set device

    # Create the model
    model_helper = models.create_model_helper(args.framework, args.dataset, args.arch)

    compression_scheduler = None

    if args.deprecated_resume:
        print('The "--resume" flag is deprecated. Please use "--resume-from=YOUR_PATH" instead.')
        if not args.reset_optimizer:
            print('If you wish to also reset the optimizer, call with: --reset-optimizer')
            args.reset_optimizer = True
        args.resumed_checkpoint_path = args.deprecated_resume

    if args.compress: