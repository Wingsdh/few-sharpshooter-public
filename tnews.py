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

from absl import app, flags

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from modeling.dqn import FewShotEnv, train_process, ClassFewShotEnv
from modeling.classifier import LabelData
from modeling.mlm_encoder import MlmBertEncoder
from modeling.retriever_classifier import RetrieverClassifier
from utils.cls_train import eval_model, dump_result

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


label_2_desc = {'news_tech': '科技',
                'news_entertainment': '娱乐',
                'news_car': '汽车',
                'news_travel': '旅游',
                'news_finance': '财经',
                'news_edu': '教育',
                'news_world': '国际',
                'news_house': '房产',
                'news_game': '电竞',
                'news_military': '军事',
                'news_story': '故事',
                'news_culture': '文化',
                'news_sports': '体育',
                'news_agriculture': '农业',
                'news_stock': '股票'}

code_2_label = {0: 'news_tech',
                1: 'news_entertainment',
                2: 'news_car',
                3: 'news_travel',
                4: 'news_finance',
                5: 'news_edu',
                6: 'news_world',
                7: 'news_house',
                8: 'news_game',
                9: 'news_military',
                10: 'news_story',
                11: 'news_culture',
                12: 'news_sports',
                13: 'news_agriculture',
                14: 'news_stock'}


def get_data_fp(use_index):
    train_fp = f'dataset/tnews/train_{use_index}.json'
    dev_fp = f'dataset/tnews/dev_{use_index}.json'
    test_fp = 'dataset/tnews/test.json'
    my_test_fp = []
    for ind in range(5):
        if str(ind) != use_index:
            my_test_fp.append(f'dataset/tnews/dev_{ind}.json')
    return train_fp, dev_fp, my_test_fp, test_fp


def main(_):
    # 参数

    # 加载数据
    train_fp, dev_fp, my_test_fp, test_fp = get_data_fp(FLAGS.c)
    key_label = 'label_desc'
    key_sentence = 'sentence'
    train_data = load_data(train_fp, key_sentence, key_label)
    data = [LabelData(text, label) for text, label in train_data]
    dev_data = load_data(dev_fp, key_sentence, key_label)
    dev_sentences = [LabelData(text, label) for text, label in dev_data]

    # 初始化encoder
    model_path = '../chinese_roberta_wwm_ext_L-12_H-768_A-12'
    weight_path = '../temp_tnews.weights'

    prefix = '以下一则关于啊啊的新闻。'
    mask_ind = [6, 7]
    encoder = MlmBertEncoder(model_path, weight_path, train_data, dev_data, prefix, mask_ind, label_2_desc, 16,
                             merge=MlmBertEncoder.CONCAT)
    n_top = len(train_data) // 10

    # fine tune
    best_acc = 0
    for epoch in range(10):
        print(f'Training epoch {epoch}')
        encoder.train(1)
        # 加载分类器
        classifier = RetrieverClassifier(encoder, data, n_top=n_top)

        print('Eval model')
        rst = eval_model(classifier, [dev_fp], key_sentence, key_label)
        if rst > best_acc:
            encoder.save()
            best_acc = rst
            print(f'Save for best {best_acc}')

        print(f'{train_fp} + {dev_fp} -> {rst}')

    encoder.load()

    # 加载最终模型
    classifier = RetrieverClassifier(encoder, data, n_top=n_top)

    # train_py_env = ClassFewShotEnv(classifier.retriever, dev_sentences, code_2_label)
    # train_py_env.reset()
    # train_process(train_py_env, train_py_env)

    # 自测试集测试
    rst = eval_model(classifier, my_test_fp, key_sentence, key_label)
    print(f'{train_fp} + {dev_fp} -> {rst}')
    # encoder.key_tokens.update(encoder.pred_char_set)
    # encoder.key_token_index = encoder.tokenizer.tokens_to_ids(encoder.key_tokens)
    # classifier = RetrieverClassifier(encoder, data, n_top=7)
    # rst = eval_model(classifier, my_test_fp, key_sentence, key_label)
    print(f'{train_fp} + {dev_fp} -> {rst}')
    print(encoder.pred_char_set)
    print(encoder.pred_char_set - encoder.key_tokens)

    # 官方测试集
    test_data = load_test_data(test_fp)
    test_data = infer(test_data, classifier)
    outp_fn = f'tnewsf_predict_{FLAGS.c.replace("few_all", "all")}.json'
    dump_result(outp_fn, test_data)


if __name__ == "__main__":
    app.run(main)
