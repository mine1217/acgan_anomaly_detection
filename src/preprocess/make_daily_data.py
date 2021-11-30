# -*- coding: utf-8 -*-
"""
data/raw/log*.csvから，1日ごとのデータを作成してdata/elect_data/にデバイスごとに保存．

Example:
    For training data.
    ::
        python2.7 src/preprocess/make_daily_data.py --train

    For detecting data.
    ::
        python2.7 src/preprocess/make_daily_data.py
"""
import os
import re
import sys
from subprocess import check_output as co
from collections import Counter
from glob import glob
from subprocess import CalledProcessError
import argparse


def transAPowers(data):
    dat = data[0].split(":")
    time = int(dat[0]) * 60 + int(dat[1])
    # =/720/1.414
    return [time, int(float(int(data[3][4:12], 16) * 2) / 1018.1944)]


def estimateHowManyMissingValues(dat1, dat2):
    ret = int((dat2 - dat1) / 12.0 + 0.1) - 1
    return ret


def complementMissingValues(dataList):
    # dataListの電力データ補完してlistで返す．
    ret = []
    count_total = 0
    count = estimateHowManyMissingValues(0, dataList[0][0])
    count_total += count
    # ret.append(dataList[0][1])
    for j in range(count):  # 0~最初の計測時間のデータ補間
        ret.append(dataList[0][0])

    for i in range(len(dataList) - 1):  # 欠損データの補間
        ret.append(dataList[i][1])
        # 補間すべきデータの数を数える
        count = estimateHowManyMissingValues(
            dataList[i][0], dataList[i + 1][0])
        count_total += count
        # print count
        for j in range(count):  # データ補間
            r = (j + 1) / float(count + 1)
            ret.append((dataList[i][1] * (1 - r) + dataList[i + 1][1] * r))

    count = estimateHowManyMissingValues(dataList[len(dataList) - 1][0], 1440)
    count_total += count
    while len(ret) < 120:  # 最後の計測時間までのデータ補間
        ret.append(dataList[len(dataList) - 1][1])
    return ret


def load_target(input_dir, train):
    if train:
        targetFileList = sorted(glob(input_dir + "log*.csv")
                                )[-1200:-1][::-1]  # 最新2年分のデータ
        save_dir = "./data/elect_data/train/"
    else:
        targetFileList = sorted(glob(input_dir + "log*.csv")
                                )[-2:-1][::-1]  # 直近1日のデータ
        save_dir = "./data/elect_data/detect/"
    return targetFileList, save_dir


def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', action='store_true',
                        help='train data or detect data')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    # 1日ごとのデータを作成する
    start = 6
    end = 18
    transFunc = transAPowers
    input_dir = "./data/raw/"
    targetFileList, save_dir = load_target(input_dir, args.train)

    devices = [i.split(",")[1] for i in co(
        ["cat", targetFileList[0]]).split("\n") if i != ""]
    print(devices)
    deviceList = Counter(devices).keys()
    print(Counter(devices))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in deviceList:  # i:現在のデバイス
        print(i)
        data = ""  # デバイス一つのデータ
        data_count = 0  # あるデバイスのデータの数
        for j in targetFileList:  # j:一日のデータ
            data_count += 1
            day_data = ""
            try:
                lines = [k for k in co(
                    ["grep", str(i), str(j)]).split("\n") if k != ""]
            except CalledProcessError as e:
                # grepで一行も出力されないときにされる処理
                # data += "\n"
                continue

            try:
                dataList = [transFunc(line.split(",")) for line in lines]
            except ValueError:
                # データの形式が違う時
                data += "\n"
                continue
            if (len(dataList) >= 0):
                day_data = ",".join(
                    map(str, complementMissingValues(dataList)))  # 1日の電力データ(文字列)
                day = j.replace(input_dir + "log",
                                "").replace(".csv", "")  # PASSと拡張子削除，つまり日付のみ
            if day_data.count(",") >= 0:
                list = day_data.split(",")
                day_data = list[0]
                for l in range(1, 120):
                    day_data += "," + list[l]
            data += day + "," + day_data + "\n"
        with open(save_dir + "{deviceName}.csv".format(deviceName=i), "w") as f:
            f.write(data)
            # print("{device}".format(device=i))


if __name__ == "__main__":
    main()
