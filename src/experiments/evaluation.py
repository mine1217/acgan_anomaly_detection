import argparse
import os

import _pathmagic
import pandas as pd
# from acanogan import acanogan_model,acanogan_predict, acanogan_test
from sklearn import metrics
import matplotlib.pyplot as plt
# import scikitplot as skplt


def load_scores(normal_files, anomaly_files):
    def load(file_path):
        return pd.read_csv(file_path,
                           index_col=0,
                           header=None,
                           names=["score"])
    normal_scores = [load(file_path) for file_path in normal_files]
    anomaly_scores = [load(file_path) for file_path in anomaly_files]
    return normal_scores, anomaly_scores


def make_roc_dataset(normal, anomaly):
    normal["y"] = 0
    anomaly["y"] = 1
    roc_dataset = pd.concat([normal, anomaly])
    return roc_dataset


def cutoff(roc_dataset, fprs, tprs, thresholds):
    """

    Args:
        fpr:
        tpr:
        thresholds:

    Returns:
        dict:fprs[cutoff_index],tprs[cutoff_index],precision,recall,specificity,f1_score,auc
    """
    # tpr-fprが最大となる点が最適な閾値
    cutoff_criterion = tprs - fprs
    cutoff_index = cutoff_criterion.argmax()

    pred_label = (roc_dataset["score"] > thresholds[cutoff_index]).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(
        roc_dataset["y"], pred_label, ).ravel()
    # recall=tp/(tp+fn)
    # precision=tp/(tp+fp)
    precision = metrics.precision_score(roc_dataset["y"], pred_label)
    recall = metrics.recall_score(roc_dataset["y"], pred_label)
    specificity = tn / (tn + fp)
    f1_score = metrics.f1_score(roc_dataset["y"], pred_label)
    auc = metrics.auc(fprs, tprs)

    cutoff_result = {"fpr": fprs[cutoff_index],
                     "tpr": tprs[cutoff_index],
                     "precision": precision,
                     "recall": recall,
                     "specificity": specificity,
                     "f1_score": f1_score,
                     "auc": auc}
    return cutoff_result


def plot_roc_curve(
        roc_curve_data,
        save_path="output/experiments/roc_curve/roc_test.png"):
    plt.figure(figsize=(9, 9))
    # plt.figure(figsize=(5, 5))
    plt.rcParams["font.size"] = 25
    plt.xlabel('FPR: False Positive Rate')
    plt.ylabel('TPR: True Positive Rate')
    plt.grid()
    plt.rcParams["font.size"] = 20
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    # for i, roc_curve_data in enumerate(roc_curve_datas):
    plt.plot(
        roc_curve_data["fprs"],
        roc_curve_data["tprs"],
        lw=2)
    # plt.legend(loc='lower right')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    # plt.show()


def make_detect_accuracy(normal_file, anomaly_file, plot_save, accuracy_save):
    """

    Args:
        normal_file:
        anomaly_file:
        plot_save:
        accuracy_save:
    """
    normal_score = pd.read_csv(normal_file,
                               index_col=0,
                               header=None,
                               names=["score"])
    anomaly_score = pd.read_csv(anomaly_file,
                                index_col=0,
                                header=None,
                                names=["score"])
    dataset = make_roc_dataset(normal_score, anomaly_score)
    fprs, tprs, thres = metrics.roc_curve(
        dataset.y.to_list(), dataset.score.to_list())
    
    print(len(dataset.y.to_list()))


    roc_curve_data = {"fprs": fprs, "tprs": tprs, "thres": thres}
    cutoff_result = cutoff(dataset, fprs, tprs, thres)

    cutoff_result_df = pd.DataFrame.from_dict(cutoff_result, orient='index').T
    os.makedirs(os.path.dirname(accuracy_save), exist_ok=True)
    cutoff_result_df.to_csv(accuracy_save, index=False)
    # print(cutoff_result)
    plot_roc_curve(roc_curve_data, plot_save)
    # roc_curve_datas.append[{"fpr": fpr, "tpr": tpr, "thres": thres}]


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--normal",
        default="output/experiments/score/acanogan/5032AB_normal.csv",
        help="Normal data score File")
    parser.add_argument(
        "-a",
        "--anomaly",
        default="output/experiments/score/acanogan/5032AB_anomaly.csv",
        help="Anomaly data score File")
    parser.add_argument(
        "-ms",
        "--roc_save",
        default="output/experiments/roc_curve/5032AB.png",
        help="File to save the roc curve")
    parser.add_argument(
        "-as",
        "--accuracy_save",
        default="output/experiments/accuracy/5032AB.csv",
        help="File to save the roc accuracy")
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    make_detect_accuracy(
        args.normal,
        args.anomaly,
        args.roc_save,
        args.accuracy_save)


if __name__ == '__main__':
    main()
