# -*- coding:utf-8 -*-
"""
  @Time : 2021-06-20 11:20
  @Authpr : MandyL
  @File : wsc_train.py
"""
import json
import os

from tqdm import tqdm

from modeling.classifier import LabelData
from modeling.retriever_classifier import RetrieverClassifier
from utils.data_utils import load_wsc_data


def train(train_fp, dev_fp, key_sentence, key_label, encoder):
    # 加载训练数据
    train_data = load_wsc_data(train_fp)

    data = [LabelData(d[key_sentence], d[key_label]) for d in train_data]

    # 加载分类器
    classifier = RetrieverClassifier(encoder, data, n_top=7)
    return classifier


def eval_wsc_model(classifier, test_fps, key_sentence, key_label, need_print=False):
    cnt = 0
    test_data = []
    for fp in test_fps:
        each_data = load_wsc_data(fp, key_sentence, key_label)
        test_data.extend(each_data)

    for sentence, label in tqdm(test_data):
        pred_label = classifier.classify(sentence)
        if label == pred_label:
            cnt += 1
        elif need_print:
            print('-----')
            print(label, pred_label)
            print(sentence)

    return cnt / len(test_data)


def dump_result(filename, data, root_path='../fewshot_train/result/'):
    with open(os.path.join(root_path, filename), 'w', encoding='utf-8') as fd:
        for d in data:
            json.dump(d, fd)
            fd.write("\n")
