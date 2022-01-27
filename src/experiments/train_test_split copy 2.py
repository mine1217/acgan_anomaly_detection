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
import jpholiday
import json

def encode_day_to_label(df: pd.DataFrame) -> dict:
    """
    日付を曜日，祝日により数値にエンコードする．[月,日]=[0,6]，祝日=7

    Args:
        df(pandas DataFrame):Daily data
    Returns:
        dict:{date\:label}
    """
    index_list = df.index.to_list()
    dates = list(map(lambda x: datetime.datetime.strptime(
        "20" + x, '%Y.%m.%d'), index_list))
    label = list(map(lambda x: x.weekday(), dates))
    holiday_label = list(map(lambda x: jpholiday.is_holiday(
        datetime.date(x.year, x.month, x.day)), dates))
    label = [7 if holiday_label[i] else label[i] for i in range(len(label))]
    return dict(zip(index_list, label))

def train_test_split(input, split_date, train_save, test_save, seed):
    """
    train_test_split

    Args:
    """
    args = arg_parse()
    data_dim = 20


    input = pd.read_csv(input, header=None, index_col=0)

    combination = json.load(open(args.combination))
    combination = list(combination.values())
    day_to_label = encode_day_to_label(input).values()
    class_labels = [combination[i] for i in day_to_label]
    index_for_class = np.empty(0)
    print("********data_size_for_class********")
    for i in np.unique(class_labels):
        print("class %d: %d" % (i, np.count_nonzero(class_labels==i)))
        index_for_class = np.append(index_for_class, np.random.choice(np.where(class_labels == i)[0], size=data_dim, replace=False))


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
        train_dates = [d.strftime("%Y.%m.%d")[2:] for i, d in enumerate(dates) if np.any(index_for_class==i)]
        test_dates = [d.strftime("%Y.%m.%d")[2:] for i, d in enumerate(dates) if not np.any(index_for_class==i)]
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
        "-c",
        "--combination",
        default="data/experiments/combination/5032AB.json",
        help="combination(best label to date) file path")
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
