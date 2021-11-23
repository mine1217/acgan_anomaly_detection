"""
AC-GANによる学習済みmodelを用いてinputの異常検知を行い結果をoutputに保存．

Example:
    5032AB example

    ::
        python3 src/acanogan/detect_anomaly.py --input data/elect_data/detect/5032AB.csv\
 --g_model models/acgan/5032AB/generator.h5 --d_model models/acgan/5032AB/discriminator.h5\
 --combination data/processed/combination/5032AB.json --minmax data/processed/minmax/5032AB.json\
 --output output/deviceState/5032AB.json
"""
import _pathmagic
import os
import json
import argparse
import numpy as np
from keras.optimizers import Adam
import pandas as pd
from src.acgan import acgan
from src.acanogan import acanogan_predict
from src.preprocess import optimize


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="data/elect_data/detect/5032AB.csv",
        help="input file path")
    parser.add_argument(
        "-gm",
        "--g_model",
        default="models/acgan/5032AB/generator.h5 ",
        help="acgan generator model file path")
    parser.add_argument(
        "-dm",
        "--d_model",
        default="models/acgan/5032AB/discriminator.h",
        help="acgan discriminator model file path")
    parser.add_argument(
        "-c",
        "--combination",
        default="data/processed/combination/5032AB.json",
        help="combination(best label to date) file path")
    parser.add_argument(
        "-mm",
        "--minmax",
        default="data/processed/minmax/5032AB.json",
        help="data minmax file path")
    parser.add_argument(
        "-t",
        "--threshold",
        default=100,
        help="threshold to be considered anomaly")
    parser.add_argument(
        "-o",
        "--output",
        default="output/deviceState/5032AB.json",
        help="output file path")
    args = parser.parse_args()
    return args


def main():
    """
    main
    """
    args = arg_parse()
    # Prepare data
    input = pd.read_csv(args.input, index_col=0, header=None)
    minmax = json.load(open(args.minmax))
    minimum, maximum = minmax["minimum"], minmax["maximum"]
    x_test = input.values

    # Prepare class label
    combination = json.load(open(args.combination))
    combination = list(combination.values())
    label = combination[next(
        iter(optimize.encode_day_to_label(input).values()))]
    num_classes = int(max(combination)) + 1

    # AC-Gan model load
    acgan_obj = acgan.ACGAN(
        num_classes=num_classes,
        minimum=minimum,
        maximum=maximum
    )
    generator = acgan_obj.generator
    discriminator = acgan_obj.discriminator

    # Data normalize,shape
    sub = maximum - minimum
    if sub == 0:
        # all 0 data
        sub = 1
    x_test = (x_test - minimum) / sub
    x_test = x_test[:, :, None]
    x_test = x_test.astype(np.float32)

    # Detect anomaly by AC-Anogan
    acanogan_optim = Adam(lr=0.001, amsgrad=True)
    generator.load_weights(args.g_model)
    discriminator.load_weights(args.d_model)
    anomaly_score, generated_data = acanogan_predict.predict(
        x_test[0], generator, discriminator, acanogan_optim, label=np.array(
            [label]), iterations=100)

    if anomaly_score < args.threshold:
        state = "stable"
    else:
        state = "unstable"
    result = {"state": state}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as w:
        json.dump(result, w, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
