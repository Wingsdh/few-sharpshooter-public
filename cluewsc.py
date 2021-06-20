# -*- coding:utf-8 -*-
"""
  @Time : 2021-06-19 13:43
  @Authpr : MandyL
  @File : cluewsc.py
"""

import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from modeling.classifier import LabelData
from modeling.mlm_encoder import MlmBertEncoder
from modeling.retriever_classifier import RetrieverClassifier
from utils.cls_train import eval_model, dump_result
from utils.wsc_train import eval_wsc_model, dump_result

sys.path.append('../')
sys.path.append('./')

from utils.data_utils import load_wsc_data


def infer(test_data, classifier):
    for d in test_data:
        sentence = d.pop('text')
        label = classifier.classify(sentence)
        d['label'] = label
    return test_data


label_2_desc = {'true': '正确', 'false': '错误'}


def get_data_fp(use_index):
    train_fp = f'dataset/cluewsc/train_{use_index}.json'
    dev_fp = f'dataset/cluewsc/dev_{use_index}.json'
    test_fp = 'dataset/cluewsc/test.json'
    my_test_fp = []
    for ind in range(5):
        if ind != use_index:
            my_test_fp.append(f'dataset/cluewsc/dev_{ind}.json')
    return train_fp, dev_fp, my_test_fp, test_fp


def main():
    # 加载数据
    train_fp, dev_fp, my_test_fp, test_fp = get_data_fp(1)
    key_label = 'label'
    key_sentence = 'text'
    train_data = load_wsc_data(train_fp, key_sentence, key_label)
    dev_data = load_wsc_data(dev_fp, key_sentence, key_label)

    # 初始化encoder
    model_path = 'pretrained_model/roberta/'
    prefix = '下面句子的指代关系正确吗？啊啊。'
    mask_ind = [13, 14]
    encoder = MlmBertEncoder(model_path, train_data, dev_data, prefix, mask_ind, label_2_desc, 16)

    # fine tune
    data = [LabelData(text, label) for text, label in train_data]
    for epoch in range(10):
        print(f'Training epoch {epoch}')
        encoder.train(1)
        # 加载分类器
        classifier = RetrieverClassifier(encoder, data, n_top=7)

        print('Evel model')
        rst = eval_wsc_model(classifier, [dev_fp], key_sentence, key_label)
        print(f'{train_fp} + {dev_fp} -> {rst}')

    # 加载最终模型
    classifier = RetrieverClassifier(encoder, data, n_top=3)

    # 自测试集测试
    rst = eval_wsc_model(classifier, my_test_fp, key_sentence, key_label)
    print(f'{train_fp} + {dev_fp} -> {rst}')

    # 官方测试集
    # test_data = load_test_data(test_fp)
    # test_data = infer(test_data, classifier)
    # dump_result('tnewsf_predict.json', test_data)


if __name__ == "__main__":
    main()
