"""
AC-GANによる学習済みmodelを用いて異常検知を行う実験用スクリプト．

Example:
    5032AB example

    Normal data experiments
    ::
        python3 src/experiments/acanogan_test.py --input data/experiments/test/5032AB.csv

    Anomaly data experiments
    ::
        python3 src/experiments/acanogan_test.py --input data/experiments/test/5032AB.csv --is_anomaly_test

    Save variety of things
    ::
        python3 src/experiments/acanogan_test.py --input data/experiments/test/5032AB.csv\
 --score_save output/experiments/score/acanogan/5032AB_normal.csv\
 --generated_save output/experiments/acanogan/generate/5032AB_normal.csv\
 --model_save output/experiments/acanogan/models/5032AB_normal/\

"""
import _pathmagic
import argparse
import json

import os
import numpy as np
import random
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
random.seed(0)

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

from keras import backend as K

tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from src.preprocess import optimize
import pandas as pd
from src.acgan import cnnClassfier
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from matplotlib import pyplot as plt
from matplotlib import cm
import umap.umap_ as umap

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="data/experiments/test/5032AB.csv",
        help="input file path(for test)")
    parser.add_argument(
        "-m",
        "--model",
        default="models/experiments/cnn/5032AB/classfier.h5",
        help="acgan generator model file path")
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
    parser.add_argument(
        "-w",
        "--w",
        type=float,
        default=0.1,
        help="parameter")
    parser.add_argument(
        "-as",
        "--accuracy_save",
        default="output/experiments/accuracy/5032AB.csv",
        help="File to save the roc accuracy")
    parser.add_argument(
        "-us",
        "--umap_save",
        default="output/experiments/umap/cnn/5032AB.png",
        help="File to save the roc accuracy")
    parser.add_argument(
        "-rs",
        "--roc_save",
        default="output/experiments/roc_curve/5032AB_cnn.png",
        help="File to save the roc curve")
    args = parser.parse_args()
    return args

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
    # plt.plot(
    #     roc_curve_data["fprs"],
    #     roc_curve_data["tprs"],
    #     lw=2)
    # plt.legend(loc='lower right')
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.savefig(save_path)
    # plt.show()

def main():
    args = arg_parse()
    # Prepare data
    input = pd.read_csv(args.input, index_col=0, header=None)
    minmax = json.load(open(args.minmax))
    minimum, maximum = minmax["minimum"], minmax["maximum"]
    x_test = input.values

    # Prepare class label
    combination = json.load(open(args.combination))
    combination = list(combination.values())
    day_to_label = optimize.encode_day_to_label(input).values()
    class_labels = [combination[i] for i in day_to_label]
    num_classes = int(max(combination)) + 1

    # AC-Gan model load
    cnn = cnnClassfier.cnnClassfier(
        num_classes=num_classes,
        minimum=minimum,
        maximum=maximum
    )
    classfier = cnn.classfier
    classfier.load_weights(args.model)

    # Data normalize,shape
    sub = maximum - minimum
    if sub == 0:
        # all 0 data
        sub = 1
    x_test = (x_test - minimum) / sub
    x_test = x_test[:, :, None]
    x_test = x_test.astype(np.float32)

    # Test classfier
    predict_class = np.empty((len(x_test), num_classes))
    i = 0
    for test_data in x_test:
        # anomaly_score, generated_data = predict(
        #     test_data, generator, discriminator, acanogan_optim, label=np.array(
        #         [label]), iterations=args.iterations, w=args.w,

        # Predict
        x = test_data[np.newaxis, :, :]
        predict = classfier.predict(x)

        # score_list.append(anomaly_score)
        predict_class[i] = predict[0]

        # generated_data = (generated_data * sub) + minimum
        # generated_data_list.append(generated_data)

        # Log
        i += 1
        print(f"data:{i}/{len(x_test)},score:{predict}")

    if num_classes == 2:
        correct_class = np.delete(label_binarize(class_labels, classes=[0,1,2]), 2, axis=1)
    else:
        correct_class = label_binarize(class_labels, classes=list(range(num_classes)))

    print(correct_class[:,0].shape)

    # roc plot
    plt.figure(figsize=(9, 9))
    plt.rcParams["font.size"] = 25
    plt.xlabel('FPR: False Positive Rate')
    plt.ylabel('TPR: True Positive Rate')
    plt.grid()
    plt.rcParams["font.size"] = 20
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

    colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
    for i in range(num_classes):
        fprs, tprs, thres = metrics.roc_curve(correct_class[:,i], predict_class[:,i])
        roc_curve_data = {"fprs": fprs, "tprs": tprs, "thres": thres}
        plt.plot(
            roc_curve_data["fprs"],
            roc_curve_data["tprs"],
            label='class'+str(i),
            color=colorlist[i],
            lw=2)
    os.makedirs(os.path.dirname(args.roc_save), exist_ok=True)
    plt.savefig(args.roc_save)


    # score
    auc = metrics.roc_auc_score(correct_class, predict_class, multi_class="ovo")

    predict_class = [np.argmax(i) for i in predict_class]
    conf = metrics.confusion_matrix(class_labels, predict_class)

    FP = conf.sum(axis=0) - np.diag(conf)  
    FN = conf.sum(axis=1) - np.diag(conf)
    TP = np.diag(conf)
    TN = conf.sum() - (FP + FN + TP)

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    specificity = TN/(TN+FP)

    fpr = zip(list(map(lambda x: "class" + str(x) + ": fpr", list(range(num_classes)))), fpr)
    tpr = zip(list(map(lambda x: "class" + str(x) + ": tpr", list(range(num_classes)))), tpr)
    specificity = zip(list(map(lambda x: "class" + str(x) + ": specificity", list(range(num_classes)))), specificity)

    acuarry = metrics.accuracy_score(class_labels, predict_class)
    precision = metrics.precision_score(class_labels, predict_class, average="macro")
    recall = metrics.recall_score(class_labels, predict_class, average="macro")
    f1_score = metrics.f1_score(class_labels, predict_class, average="macro")
    print(acuarry)
    result = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "acuarry": acuarry,
        "auc": auc
        }
    result.update(fpr)
    result.update(tpr)
    result.update(specificity)
    result_df = pd.DataFrame.from_dict(result, orient='index').T
    result_df.to_csv(args.accuracy_save, index=False)

    mapper = umap.UMAP(random_state=0)
    embedding = mapper.fit_transform(input.values)

    embedding_x = embedding[:, 0]
    embedding_y = embedding[:, 1]

    plt.figure(dpi=500)

    for n in np.unique(predict_class):
        plt.scatter(embedding_x[predict_class == n],
                    embedding_y[predict_class == n],
                    label=n)
    
    plt.grid()
    plt.legend()
    plt.savefig(args.umap_save+"predict.png")

    plt.figure(dpi=500)
    
    for n in np.unique(class_labels):
        plt.scatter(embedding_x[class_labels == n],
                    embedding_y[class_labels == n],
                    label=n)

    plt.grid()
    plt.legend()
    plt.savefig(args.umap_save+"correct.png")


if __name__ == '__main__':
    main()
