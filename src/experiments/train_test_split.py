"""
train_test_split
date以降の日のデータをtestデータとする．

Example:
    503342 example

    ::
        python3 src/experiments/train_test_split.py --input data/experiments/all/503342.csv --date 2020.01.01\
 --train_save data/experiments/train/503342.csv --test data/experiments/test/503342.csv
"""
import argparse
import os
import pandas as pd
import numpy as np
import datetime


def train_test_split(input, split_date, train_save, test_save, seed):
    """
    train_test_split

    Args:
    """
    input = pd.read_csv(input, header=None, index_col=0)
    os.makedirs(os.path.dirname(train_save), exist_ok=True)
    os.makedirs(os.path.dirname(test_save), exist_ok=True)
    # split_date = date[2:]
    dates = [
        datetime.datetime.strptime(
            f"20{d}",
            "%Y.%m.%d") for d in input.index.to_list()]

    np.random.seed(seed)

    if split_date is None:
        rand = np.random.randint(1, 4, len(dates))
        print(rand)
        train_dates = [d.strftime("%Y.%m.%d")[2:] for i, d in enumerate(dates) if not 3 == rand[i]]
        test_dates = [d.strftime("%Y.%m.%d")[2:] for i, d in enumerate(dates) if 3 == rand[i]]
    else:
        split_date = datetime.datetime.strptime(
            split_date,
            "%Y.%m.%d")
        train_dates = [d.strftime("%Y.%m.%d")[2:] for d in dates if split_date > d]
        test_dates = [d.strftime("%Y.%m.%d")[2:] for d in dates if split_date <= d]
    
    train = input.loc[train_dates]
    test = input.loc[test_dates]
    train.to_csv(train_save, header=False)
    test.to_csv(test_save, header=False)

def serial(s):
    end = s
    start = datetime.datetime(1899, 12, 31)
    delta = end - start
    return delta.days


def arg_parse():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        '--input',
        '-i',
        default="data/experiments/all/503342.csv",
        type=str,
        help='input csv path')
    parser.add_argument(
        '--date',
        '-d',
        default=None,
        type=str,
        help='split date')
    parser.add_argument(
        '--train_save',
        '-trs',
        default="data/experiments/train/503342.csv",
        type=str,
        help='train save csv path')
    parser.add_argument(
        '--test_save',
        '-tes',
        default="data/experiments/test/503342.csv",
        type=str,
        help='test save csv path')
    parser.add_argument(
        '--seed',
        '-sd',
        default=0,
        type=int,
        help='random_seed')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    train_test_split(args.input, args.date, args.train_save, args.test_save, args.seed)


if __name__ == '__main__':
    main()
