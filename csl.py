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
from utils.cls_train import eval_model, dump_result, eval_model_csl

sys.path.append('../')
sys.path.append('./')

from utils.data_utils import load_data, load_test_data, load_csl_keyword


def infer(test_data, classifier):
    for d in test_data:
        sentence = d.pop('sentence')
        label = classifier.classify(sentence)
        d['label'] = label
    return test_data


label_2_desc = {"0": "不能", "1":"可以"}


def get_data_fp(use_index):
    train_fp = f'dataset/csl/train_{use_index}.json'
    dev_fp = f'dataset/csl/dev_{use_index}.json'
    test_fp = 'dataset/csl/test.json'
    my_test_fp = []
    for ind in range(5):
        if ind != use_index:
            my_test_fp.append(f'dataset/csl/dev_{ind}.json')
    return train_fp, dev_fp, my_test_fp, test_fp


def main():
    # 参数

    # 加载数据
    train_fp, dev_fp, my_test_fp, test_fp = get_data_fp(0)
    key_label = 'label'
    key_sentence = 'abst'
    key_word='keyword'
    train_data = load_csl_keyword(train_fp, key_sentence, key_label,key_word)
    dev_data = load_csl_keyword(dev_fp, key_sentence, key_label,key_word)

    # 初始化encoder
    model_path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12'
    prefix = '黴鹹用'
    mask_ind = [0, 1]



    encoder = MlmBertEncoder(model_path, train_data, dev_data, prefix, mask_ind, label_2_desc, 16)

    # fine tune
    data = [LabelData(text, label) for text, label in train_data]
    for epoch in range(20):
        print(f'Training epoch {epoch}')
        encoder.train(1)
        # 加载分类器
        classifier = RetrieverClassifier(encoder, data, n_top=7)

        print('Evel model')
        rst = eval_model(classifier, [dev_fp], key_sentence, key_label)   #rst=预测的准确率
        print(f'{train_fp} + {dev_fp} -> {rst}')

    # 加载最终模型
    classifier = RetrieverClassifier(encoder, data, n_top=3)

    # 自测试集测试
    rst = eval_model_csl(classifier, my_test_fp, key_sentence, key_label,key_word)
    print(f'{train_fp} + {dev_fp} -> {rst}')

    # 官方测试集
    # test_data = load_test_data(test_fp)
    # test_data = infer(test_data, classifier)
    # dump_result('tnewsf_predict.json', test_data)


if __name__ == "__main__":
    main()
