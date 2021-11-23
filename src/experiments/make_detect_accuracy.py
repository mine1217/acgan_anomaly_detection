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
                     "thresholds": thresholds[cutoff_index],
                     "precision": precision,
                     "recall": recall,
                     "specificity": specificity,
                     "f1_score": f1_score,
                     "auc": auc}
    return cutoff_result


def plot_roc_curve(
        roc_curve_datas,
        labels,
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
    for i, roc_curve_data in enumerate(roc_curve_datas):
        plt.plot(
            roc_curve_data["fprs"],
            roc_curve_data["tprs"],
            lw=2,
            label=labels[i])
    plt.legend(loc='lower right')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    # plt.show()


def make_detect_accuracy(normal_files, anomaly_files, labels, plot_save):
    """

    Args:
        normal_files:
        anomaly_files:
        labels:
        plot_save:
    Returns:

    """
    normal_scores, anomaly_scores = load_scores(normal_files, anomaly_files)
    roc_datasets = [
        make_roc_dataset(
            normal, anomaly) for normal, anomaly in zip(
            normal_scores, anomaly_scores)]
    cutoff_results = []
    roc_curve_datas = []
    for dataset in roc_datasets:
        fprs, tprs, thres = metrics.roc_curve(
            dataset.y.to_list(), dataset.score.to_list())
        roc_curve_datas.append({"fprs": fprs, "tprs": tprs, "thres": thres})
        cutoff_result = cutoff(dataset, fprs, tprs, thres)
        print(cutoff_result)
        cutoff_results.append(cutoff_result)
    plot_roc_curve(roc_curve_datas, labels, plot_save)
    # roc_curve_datas.append[{"fpr": fpr, "tpr": tpr, "thres": thres}]


def main():
    normal_files = [
        "./output/experiments/score/acanogan/5032AB_test_normal_w=0.csv",
        "./output/experiments/score/acanogan/5032AB_test_normal_w=0.1.csv",
        "./output/experiments/score/acanogan/5032AB_test_normal_w=0.5.csv",
        "./output/experiments/score/acanogan/5032AB_test_normal_w=1.csv"
    ]
    anomaly_files = [
        "./output/experiments/score/acanogan/5032AB_test_anomaly_w=0.csv",
        "./output/experiments/score/acanogan/5032AB_test_anomaly_w=0.1.csv",
        "./output/experiments/score/acanogan/5032AB_test_anomaly_w=0.5.csv",
        "./output/experiments/score/acanogan/5032AB_test_anomaly_w=1.csv"
    ]
    plot_save = "output/experiments/roc_curve/roc_acanogan.png"
    # normal_files = [
    #     "./output/experiments/score/anogan/5032AB_test_normal_w=0.csv",
    #     "./output/experiments/score/anogan/5032AB_test_normal_w=0.1.csv",
    #     "./output/experiments/score/anogan/5032AB_test_normal_w=0.5.csv",
    #     "./output/experiments/score/anogan/5032AB_test_normal_w=1.csv"]
    # anomaly_files = [
    #     "./output/experiments/score/anogan/5032AB_test_anomaly_w=0.csv",
    #     "./output/experiments/score/anogan/5032AB_test_anomaly_w=0.1.csv",
    #     "./output/experiments/score/anogan/5032AB_test_anomaly_w=0.5.csv",
    #     "./output/experiments/score/anogan/5032AB_test_anomaly_w=1.csv"]
    # plot_save = "output/experiments/roc_curve/roc_anogan.png"
    labels = [
        r"$\lambda=0$",
        r"$\lambda=0.1$",
        r"$\lambda=0.5$",
        r"$\lambda=1$"
    ]
    # normal_files = [
    #     "./output/experiments/score/acanogan/5032AB_30_normal.csv",
    #     "./output/experiments/score/acanogan/5032AB_60_normal.csv",
    #     "./output/experiments/score/acanogan/5032AB_90_normal.csv"]
    # anomaly_files = [
    #     "./output/experiments/score/acanogan/5032AB_30_anomaly.csv",
    #     "./output/experiments/score/acanogan/5032AB_60_anomaly.csv",
    #     "./output/experiments/score/acanogan/5032AB_90_anomaly.csv"]
    # plot_save = "output/experiments/roc_curve/roc_data_num.png"
    # labels = ["data=30",
    #           "data=60",
    #           "data=90"]
    make_detect_accuracy(normal_files, anomaly_files, labels, plot_save)


if __name__ == '__main__':
    main()
