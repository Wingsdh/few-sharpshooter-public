# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    tnews.py
   Description :
   Author :       Wings DH
   Time：         6/16/21 10:40 PM
-------------------------------------------------
   Change Activity:
                   6/16/21: Create
-------------------------------------------------
"""

import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from modeling.classifier import LabelData
from modeling.mlm_encoder import MlmBertEncoder
from modeling.retriever_classifier import RetrieverClassifier
from utils.cls_train import eval_model, dump_result
from absl import app, flags

sys.path.append('../')
sys.path.append('./')

from utils.data_utils import load_data, load_test_data

flags.DEFINE_string('c', '0', 'index of tnews dataset')
FLAGS = flags.FLAGS


def infer(test_data, classifier):
    for d in test_data:
        sentence = d.pop('sentence')
        label = classifier.classify(sentence)
        d['label'] = label
    return test_data


label_2_desc = {'Positive': '开心',
                'Negative': '失望'}


def get_data_fp(use_index):
    train_fp = f'dataset/eprstmt/train_{use_index}.json'
    dev_fp = f'dataset/eprstmt/dev_{use_index}.json'
    test_fp = 'dataset/eprstmt/test.json'
    my_test_fp = []
    for ind in range(5):
        if ind != use_index:
            my_test_fp.append(f'dataset/eprstmt/dev_{ind}.json')
    return train_fp, dev_fp, my_test_fp, test_fp


def main(_):
    # 参数

    # 加载数据
    train_fp, dev_fp, my_test_fp, test_fp = get_data_fp(0)
    key_label = 'label'
    key_sentence = 'sentence'
    train_data = load_data(train_fp, key_sentence, key_label)
    dev_data = load_data(dev_fp, key_sentence, key_label)
    data = [LabelData(text, label) for text, label in train_data]

    # 初始化encoder
    model_path = '../chinese_roberta_wwm_ext_L-12_H-768_A-12'
    weight_path = '../temp_eprstmt.weights'

    prefix = '我很啊啊！'
    mask_ind = [2, 3]
    encoder = MlmBertEncoder(model_path, weight_path, train_data, dev_data, prefix, mask_ind, label_2_desc, 4)

    # fine tune
    best_acc = 0
    for epoch in range(10):
        print(f'Training epoch {epoch}')
        encoder.train(1)
        # 加载分类器
        classifier = RetrieverClassifier(encoder, data, n_top=3)

        print('Eval model')
        rst = eval_model(classifier, [dev_fp], key_sentence, key_label)
        print(f'{train_fp} + {dev_fp} -> {rst}')
        if rst > best_acc:
            encoder.save()
            best_acc = rst
            print(f'Save for best {best_acc}')

    # 加载最终模型
    encoder.load()
    classifier = RetrieverClassifier(encoder, data, n_top=3)

    # 自测试集测试
    rst = eval_model(classifier, my_test_fp, key_sentence, key_label)
    print(f'{train_fp} + {dev_fp} -> {rst}')

    # 官方测试集
    test_data = load_test_data(test_fp)
    test_data = infer(test_data, classifier)
    outp_fn = f'eprstmt_predict_{FLAGS.c.replace("few_all", "all")}.json'
    dump_result(outp_fn, test_data)


if __name__ == "__main__":
    app.run(main)
