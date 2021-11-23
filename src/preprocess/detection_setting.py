"""
data/raw/deviceList.json中のデバイスIDのデータの中で，全て0のデータとmodels/acgan/にモデルがないデータ以外，
data/elect_data/detect/からdata/processed/detect/にデータをコピー．
また，deviceList中のdeviceIdの結果保存用ファイル作成．

Example:
    Usage
    ::
        python3 src/preprocess/detection_setting.py
"""
import json
import os
import shutil

import pandas as pd


def preprocess_for_detection():
    """
    異常判定のためのプロジェクト構成の設定を行う．
    """
    with open("data/raw/deviceList.json") as f:
        device_list_json = json.load(f)
    device_list = device_list_json["list"]

    model_path = "models/acgan/"
    model_list = os.listdir(model_path)
    is_model_device_list = list(set(device_list) & set(model_list))
    is_model_device_data = pd.concat(
        pd.read_csv(
            "data/elect_data/detect/{}.csv".format(device),
            header=None) for device in is_model_device_list)

    del is_model_device_data[0]
    is_model_device_data["device"] = is_model_device_list
    is_model_device_data = is_model_device_data.set_index("device")

    # Except on days when all elements < 1.
    detect_device_list = is_model_device_data[is_model_device_data > 1].dropna(
        how='all').index.to_list()

    detect_dir = "data/processed/detect/"
    os.makedirs(detect_dir, exist_ok=True)
    for device in detect_device_list:
        shutil.copy(
            "data/elect_data/detect/{}.csv".format(device),
            detect_dir)

    # detectしないもの
    except_device_list = list(set(device_list) ^ set(detect_device_list))
    result = {"state": "stable"}
    os.makedirs("output/deviceState/", exist_ok=True)
    for device in except_device_list:
        with open("output/deviceState/{}.json".format(device), "w") as w:
            json.dump(result, w, indent=4, ensure_ascii=False)


def main():
    preprocess_for_detection()


if __name__ == '__main__':
    main()
