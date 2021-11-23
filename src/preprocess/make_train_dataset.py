"""
inputから学習用データセットを作成して，outputに出力する．

Example:
    5032AB example
    ::
        python3 src/preprocess/make_train_dataset.py --input data/elect_data/train/5032AB.csv\
 --output data/processed/train/5032AB.csv
"""
import argparse
import os
import pandas as pd


def except_outlier(df: pd.DataFrame) -> pd.DataFrame:
    """
    dfから全ての要素が1未満の日を除く．また，いずれかの要素がμ＋4σより大きい日を除く．

    Args:
        df(pandas DataFrame):Input data

    Returns:
        pandas DataFrame:Excepted data
    """
    average = df.values.flatten().mean()
    sd = df.values.flatten().std()
    # Except on days when all elements < 1.
    date = df[df > 1].dropna(how='all').index.to_list()
    df = df.loc[date]
    outlier_max = average + sd * 4
    # Except on days when any elements > μ + 4σ.
    df = df[df <= outlier_max].dropna(how='any')
    return df


def make_dataset(input: str, output: str):
    """
    inputから学習用データセットを作成する．

    Args:
        input(str):Input csv path
        output(str):Output csv path
    """
    save_dir = os.path.dirname(output)
    input = pd.read_csv(input, header=None, index_col=0)
    dataset = except_outlier(input)
    os.makedirs(save_dir, exist_ok=True)
    if len(dataset) >= 30:
        dataset.to_csv(output, header=False)


def arg_parse():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        '--input',
        '-i',
        default="data/elect_data/train/5032AB.csv",
        type=str,
        help='input csv path')
    parser.add_argument(
        '--output',
        '-o',
        default="data/processed/train/5032AB.csv",
        type=str,
        help='output csv path')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    make_dataset(args.input, args.output)


if __name__ == '__main__':
    main()
