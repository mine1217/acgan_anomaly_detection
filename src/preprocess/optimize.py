"""
GMMによるクラスタリングを用いて，最適な曜日の組み合わせを求めて，save，combination_saveに保存する．\
例えば，月水金のデータをクラス A，火木をB，土日祝日をCのように3クラスにラベル付を行う．

Example:
    5032AB example
    ::
        python3 src/preprocess/optimize.py --input data/processed/train/5032AB.csv\
 --save data/processed/label/5032AB.csv --combination_save data/processed/combination/5032AB.json
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import datetime
import jpholiday
from sklearn.mixture import GaussianMixture
from sklearn.metrics import log_loss
import category_encoders as ce


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


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min max normalization.

    Args:
        df(pandas DataFrame):Input data
    Returns:
        pandas DataFrame:Normalized data
    """
    minimum = df.min().min()
    maximum = df.max().max()
    if (maximum - minimum) == 0:
        return df - df
    return (df - minimum) / (maximum - minimum)


def gmm_clustering(df: pd.DataFrame, n_clusters: int = 2, seed: int = 0) -> pd.DataFrame:
    """
    GMMによるクラスタリングを行い各クラスタへの所属確率を返す．

    Args:
        df(pandas DataFrame):Input data
        n_clusters(int):クラスタ数．
    Returns:
        pandas DataFrame:各クラスタへの所属確率
    """
    gmm = GaussianMixture(n_components=n_clusters, max_iter=20, random_state=seed)
    gmm.fit(df)
    prob = gmm.predict_proba(df)
    columns = [str(i) for i in range(0, len(prob[0]))]
    prob_df = pd.DataFrame(prob, columns=columns, index=df.index)
    return prob_df


def grouping_class(label: dict, prob: pd.DataFrame) -> pd.DataFrame:
    """
    labelの各クラス（数値）ごとに最も所属確率の高いクラスタに割り当てることで，クラスの組み合わせを求める．

    Args:
        label(dict):encode_day_to_labelによる，日付ごとの数値．
        prob(pandas DataFrame):GMM clusteringによって得られる各クラスタへの所属確率．
    Returns:
        pandas DataFrame:OneHot表現のクラスの組み合わせ．
    """
    class_label_df = pd.DataFrame(
        label.values(),
        columns=["label"],
        index=prob.index)
    for i in range(8):
        # i label dates
        dates = [k for k, v in label.items() if v == i]
        label_i_prob = prob.loc[dates]
        # 最も所属確率の高いクラスタでlabel i更新
        class_label_df[class_label_df.label == i] = label_i_prob.sum().idxmax()
    # one hot encode
    ce_one = ce.OneHotEncoder(cols=["label"], handle_unknown='impute')
    class_label_df = ce_one.fit_transform(class_label_df)
    return class_label_df


def make_best_class_label(input: str, save: str, combination_save: str, no_gmm: bool, seed: int = 0):
    """
    最適なクラスの組み合わせを探索して，それをクラスラベルとして保存する．
    また，いずれのクラスタ数においても，クラスが属していないクラスタが存在している場合は，適切な組み合わせなしとする．

    Args:
        input(str):Input csv path
        save(str):Save label csv path
        combination_save(str):Save combination json path
    """
    input = pd.read_csv(input, header=None, index_col=0)

    # Encode label
    label = encode_day_to_label(input)

    # normalization
    input = normalize(input)
    best_loss = -1
    best_class_label = pd.DataFrame(
        [1] * len(input),
        columns=["label"],
        index=input.index)

    if no_gmm:
        for c in range(3, 8):
            prob_df = gmm_clustering(input, n_clusters=c, seed=seed)
            best_class_label = grouping_class(label, prob_df)
            print(c)
            if len(best_class_label.columns) == 3:
                break
    else:
        for c in range(2, 8):
            # Clustering
            prob_df = gmm_clustering(input, n_clusters=c, seed=seed)

            # # Grouping to the highest probability cluster.
            class_label = grouping_class(label, prob_df)
            

            # There is a cluster that does not have any labels.
            if len(class_label.columns) < len(prob_df.columns):
                continue

            # # Calc cross entropy loss
            loss = log_loss(class_label, prob_df)

            # # Update best
            if loss < best_loss or best_loss < 0:
                best_loss = loss
                best_class_label = class_label

    # Translate best_class_label from one hot to number
    best_class_label = pd.DataFrame(
        np.argmax(
            best_class_label.values,
            axis=1),
        index=best_class_label.index)
    # Save
    dir_name = os.path.dirname(save)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)
    best_class_label.to_csv(save, header=False, index=True, mode="w")
    save_combinations(label, best_class_label, combination_save)


def save_combinations(
        label: dict,
        best_class_label: pd.DataFrame,
        combination_save: str):
    """
    best_class_labelの組み合わせをcombination_saveにJson形式で保存．

    Args:
        label:encode_day_to_labelによる，日付ごとの数値．
        best_class_label:OneHot形式のクラスラベル
        combination_save:Save combination json path
    """
    best_combinations_list = [0] * 8
    for week_num in range(8):
        for date, best_label in best_class_label.iterrows():
            if label[date] == week_num:
                best_combinations_list[week_num] = int(best_label[0])
                break
    key_list = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "holiday"]
    best_combinations = dict(zip(key_list, best_combinations_list))
    dir_name = os.path.dirname(combination_save)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)
    json.dump(
        best_combinations,
        open(
            combination_save,
            'w'),
        ensure_ascii=False,
        indent=4)


def arg_parse():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        '--input',
        '-i',
        default="data/processed/train/5032AB.csv",
        type=str,
        help='input csv path')
    parser.add_argument(
        '--save',
        '-s',
        default="data/processed/label/5032AB.csv",
        type=str,
        help='save label csv path')
    parser.add_argument(
        '--combination_save',
        '-cs',
        default="data/processed/combination/5032AB.json",
        type=str,
        help='save combination json path')
    parser.add_argument(
        '--seed',
        '-sd',
        default=0,
        type=int,
        help='random_seed')
    parser.add_argument(
        "-ng",
        "--no_gmm",
        action='store_true',
        help="fixed number of classes to don't use gmm")
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    make_best_class_label(args.input, args.save, args.combination_save, args.no_gmm, args.seed)


if __name__ == '__main__':
    main()
