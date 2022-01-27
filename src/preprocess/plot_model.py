import _pathmagic
import argparse
import json

from keras.models import Model
from keras.layers import *
from keras.utils import plot_model
from src.acgan import acgan
from src.acgan import cgan
from src.acgan import gan


def main():
    args = arg_parse()


    acgan_obj = acgan.ACGAN(
        num_classes=2,
        minimum=0,
        maximum=100
    )
    generator = acgan_obj.generator
    discriminator = acgan_obj.discriminator

    plot_model(generator, to_file=args.save + 'generator.png')
    plot_model(discriminator, to_file=args.save + 'discriminator.png')

def arg_parse():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        "-s",
        "--save",
        default="output/experiments/models/",
        help="File to save the roc curve")
    parser.add_argument(
        "-c",
        "--combination",
        default="data/experiments/combination/5032AB.json",
        help="combination(best label to date) file path")
    parser.add_argument(
        "-mm",
        "--minmax",
        default="data/experiments/minmax/5032AB.json",
        help="data minmax file path")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()