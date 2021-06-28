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
import json
import sys
import os

from tqdm import tqdm

from utils.seed import set_seed

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
set_seed()

from absl import flags, app
from modeling.classifier import LabelData
from modeling.mlm_encoder import MlmBertEncoder
from modeling.retriever_classifier import RetrieverClassifier
from utils.cls_train import dump_result

flags.DEFINE_string('c', '0', 'index of dataset')
FLAGS = flags.FLAGS

sys.path.append('../')
sys.path.append('./')


def load_csl_keyword(fp, key_sentence, key_label, keyword):
    data = []
    with open(fp, 'r', encoding='utf-8') as fd:
        for l in fd:
            d = json.loads(l.strip())
            keywords = d[keyword]
            for kw in keywords:
                data.append((kw + '概括以下文章，' + d[key_sentence], d[key_label]))

        # list_keyword = ['，'.join(da[keyword]) for da in data]
        # data = [(str_keyword + '概括' + d[key_sentence], d[key_label]) for d, str_keyword in zip(data, list_keyword)]
        print(f'Loaded {len(data)} data from {fp}')
    return data


def load_csl_eval_keyword(fp, key_sentence, key_label, keyword):
    data = []
    with open(fp, 'r', encoding='utf-8') as fd:
        for l in fd:
            d = json.loads(l.strip())
            keywords = d[keyword]
            each_data = []
            for kw in keywords:
                each_data.append((kw + '概括以下文章，' + d[key_sentence], d[key_label]))
            data.append(each_data)

        # list_keyword = ['，'.join(da[keyword]) for da in data]
        # data = [(str_keyword + '概括' + d[key_sentence], d[key_label]) for d, str_keyword in zip(data, list_keyword)]
        print(f'Loaded {len(data)} data from {fp}')
    return data


def eval_model_csl(classifier, test_fps, key_sentence, key_label, key_word, need_print=False):
    cnt = 0
    test_data = []
    for fp in test_fps:
        each_data = load_csl_eval_keyword(fp, key_sentence, key_label, key_word)
        test_data.extend(each_data)

    for each_data in tqdm(test_data):
        for sentence, label in each_data:
            pred_label, _ = classifier.classify(sentence)
            if pred_label == '0':
                pred_label = '0'
                break
        else:
            pred_label = '1'

        if label == pred_label:
            cnt += 1
        elif need_print:
            print('-----')
            print(label, pred_label)
            print(sentence)

    return cnt / len(test_data)  # 返回预测的准确率


def infer(test_fp, classifier, key_sentence, key_label, keyword):
    data = []
    with open(test_fp, 'r', encoding='utf-8') as fd:
        for l in fd:
            d = json.loads(l.strip())
            data.append(d)

    for d in data:
        keywords = d.pop(keyword)
        sent = d.pop(key_sentence)
        for kw in keywords:
            sentence = kw + '概括以下文章，' + sent
            label, _ = classifier.classify(sentence)
            if label == '0':
                d[key_label] = '0'
                break
        else:
            d[key_label] = '1'

    return data


label_2_desc = {"0": "绝对不能", "1": "完全可以"}


def get_data_fp(use_index):
    train_fp = f'dataset/csl/train_{use_index}.json'
    dev_fp = f'dataset/csl/dev_{use_index}.json'
    test_fp = 'dataset/csl/test.json'
    my_test_fp = []
    for ind in range(5):
        if str(ind) != use_index:
            my_test_fp.append(f'dataset/csl/dev_{ind}.json')
    return train_fp, dev_fp, my_test_fp, test_fp


def main(_):
    # 参数

    # 加载数据
    train_fp, dev_fp, my_test_fp, test_fp = get_data_fp(FLAGS.c)
    key_label = 'label'
    key_sentence = 'abst'
    key_word = 'keyword'
    train_data = load_csl_keyword(train_fp, key_sentence, key_label, key_word)
    dev_data = load_csl_eval_keyword(dev_fp, key_sentence, key_label, key_word)

    # 初始化encoder
    model_path = '../chinese_roberta_wwm_ext_L-12_H-768_A-12'
    weight_path = '../temp_csl.weights'
    prefix = '锟锟锟锟用'
    mask_ind = [0, 1, 2, 3]

    encoder = MlmBertEncoder(model_path, weight_path, train_data, dev_data, prefix, mask_ind, label_2_desc, 8,
                             merge=MlmBertEncoder.CONCAT, norm=False, lr=1e-4, policy='attention')

    # fine tune
    best_acc = 0
    data = [LabelData(text, label) for text, label in train_data]
    n_top = 5
    for epoch in range(10):
        print(f'Training epoch {epoch}')
        encoder.train(1)
        # 加载分类器
        classifier = RetrieverClassifier(encoder, data, n_top=n_top)

        print('Evel model')
        rst = eval_model_csl(classifier, [dev_fp], key_sentence, key_label, key_word)  # rst=预测的准确率
        if rst > best_acc:
            encoder.save()
            best_acc = rst
            print(f'Save for best {best_acc}')

        print(f'{train_fp} + {dev_fp} -> {rst}')

    # 加载最终模型
    encoder.load()
    classifier = RetrieverClassifier(encoder, data, n_top=n_top)

    # 自测试集测试
    rst = eval_model_csl(classifier, my_test_fp, key_sentence, key_label, key_word)
    print(f'{train_fp} + {dev_fp} -> {rst}')

    # 官方测试集
    test_data = infer(test_fp, classifier, key_sentence, key_label, key_word)

    outp_fn = f'cslf_predict_{FLAGS.c.replace("few_all", "all")}.json'
    dump_result(outp_fn, test_data)


if __name__ == "__main__":
    app.run(main)
