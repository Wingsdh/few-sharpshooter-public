# -*- coding:utf-8 -*-
"""
  @Time : 2021-06-20 12:33
  @Authpr : MandyL
  @File : bustm.py
"""

import sys
import os
import json
from tqdm import tqdm

from utils.seed import set_seed

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from modeling.classifier import LabelData
from modeling.mlm_encoder import MlmBertEncoder
from modeling.retriever_classifier import RetrieverClassifier
from utils.data_utils import load_test_data
from absl import app, flags

sys.path.append('../')
sys.path.append('./')

label_2_desc = {'0': '不同', '1': '一样'}
flags.DEFINE_string('c', '0', 'index of dataset')
FLAGS = flags.FLAGS
set_seed()


def load_bustm_data(fp, key_sentence_1, key_sentence_2, key_label):
    data = []
    with open(fp, 'r', encoding='utf-8') as fd:
        for d in fd:
            d = json.loads(d.strip())
            sentence_1 = d[key_sentence_1]
            sentence_2 = d[key_sentence_2]
            label = d[key_label]
            sentence = sentence_1 + "和" + sentence_2 + "意思锟锟"
            data.append((sentence, label))
        return data


def infer(test_data, classifier):
    for d in test_data:
        sentence_1 = d.pop('sentence1')
        sentence_2 = d.pop('sentence2')
        sentence = sentence_1 + '和' + sentence_2 + '意思锟锟'
        label = classifier.classify(sentence)
        d['label'] = label
    return test_data


def get_data_fp(use_index):
    train_fp = f'dataset/bustm/train_{use_index}.json'
    dev_fp = f'dataset/bustm/dev_{use_index}.json'
    test_fp = 'dataset/bustm/test.json'
    my_test_fp = []
    for ind in range(5):
        if ind != use_index:
            my_test_fp.append(f'dataset/bustm/dev_{ind}.json')
    return train_fp, dev_fp, my_test_fp, test_fp


def eval_bustm_model(classifier, test_fps, key_sentence_1, key_sentence_2, key_label, need_print=False):
    cnt = 0
    test_data = []
    for fp in test_fps:
        each_data = load_bustm_data(fp, key_sentence_1, key_sentence_2, key_label)
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


def main(_):
    # 加载数据
    train_fp, dev_fp, my_test_fp, test_fp = get_data_fp(FLAGS.c)
    key_label = 'label'
    key_sentence_1 = 'sentence1'
    key_sentence_2 = 'sentence2'
    train_data = load_bustm_data(train_fp, key_sentence_1, key_sentence_2, key_label)
    dev_data = load_bustm_data(dev_fp, key_sentence_1, key_sentence_2, key_label)

    # 初始化encoder
    model_path = '../chinese_roberta_wwm_ext_L-12_H-768_A-12'
    weight_path = '../temp_bustm.weights'

    prefix = ''
    mask_ind = []
    encoder = MlmBertEncoder(model_path, weight_path, train_data, dev_data, prefix, mask_ind, label_2_desc, 16,
                             norm=True)

    # fine tune
    best_acc = 0
    data = [LabelData(text, label) for text, label in train_data]
    for epoch in range(10):
        print(f'Training epoch {epoch}')
        encoder.train(1)
        # 加载分类器
        classifier = RetrieverClassifier(encoder, data, n_top=7)

        print('Eval model')
        rst = eval_bustm_model(classifier, [dev_fp], key_sentence_1, key_sentence_2, key_label)
        print(f'{train_fp} + {dev_fp} -> {rst}')
        if rst > best_acc:
            encoder.save()
            best_acc = rst
            print(f'Save for best {best_acc}')

    # 加载最终模型
    encoder.load()
    classifier = RetrieverClassifier(encoder, data, n_top=7)

    # 自测试集测试
    rst = eval_bustm_model(classifier, my_test_fp, key_sentence_1, key_sentence_2, key_label)
    print(f'{train_fp} + {dev_fp} -> {rst}')

    # 官方测试集
    test_data = load_test_data(test_fp)
    test_data = infer(test_data, classifier)
    outp_fn = f'bustm_predict_{FLAGS.c.replace("few_all", "all")}.json'
    dump_result(outp_fn, test_data)


if __name__ == "__main__":
    app.run(main)
