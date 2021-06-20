# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    mlm_encoder.py
   Description :
   Author :       Wings DH
   Time：         6/14/21 11:56 AM
-------------------------------------------------
   Change Activity:
                   6/14/21: Create
-------------------------------------------------
"""
import re
from enum import Enum
import os

from bert4keras.models import build_transformer_model, keras, K, Loss
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding, to_array
from bert4keras.tokenizers import Tokenizer

import numpy as np

from modeling.base_encoder import BaseEncoder


def random_masking(token_ids, tokenizer):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    """

    def __init__(self, data, batch_size, tokenizer, maxlen, prefix, mask_idxes, labels):
        super(data_generator, self).__init__(data, batch_size)

        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.mask_idxes = mask_idxes
        self.labels = labels
        self.prefix = prefix
        self.flag = True if not mask_idxes else False

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            if label != 2:  # label是两个字的文本
                text = self.prefix + text  # 拼接文本
            token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.maxlen)
            if random:
                source_ids, target_ids = random_masking(token_ids, self.tokenizer)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]

            if label != 2:  # label是两个字的文本
                if self.flag:
                    self.mask_indxes = [index for index, t_ids in enumerate(token_ids) if t_ids == 7233]
                # label_ids: [1093, 689]。 e.g. [101, 1093, 689, 102] =[CLS,农,业,SEP]. tokenizer.encode(label): ([101, 1093, 689, 102], [0, 0, 0, 0])
                label_ids = self.tokenizer.encode(self.labels[label])[0][1:-1]
                for i, label_id_ in zip(self.mask_idxes, label_ids):
                    # i: 7(mask1的index) ;j: 1093(农); i:8 (mask2的index) ;j: 689(业)
                    source_ids[i] = self.tokenizer._token_mask_id
                    target_ids[i] = label_id_

            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)

            if len(batch_token_ids) == self.batch_size or is_end:  # 分批padding和生成
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [batch_token_ids, batch_segment_ids, batch_output_ids], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分。作用就是只计算目标位置的loss，忽略其他位置的loss。 """

    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs  # y_true:[batch_size, sequence_length]。应该是one-hot的表示，有一个地方为1，其他地方为0：[0,0,1,...0]
        # y_mask是一个和y_true一致的shape. 1的值还为1.0，0的值还为0.0.即[0.0,0.0,1.0,...0.0]。
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        # sparse_categorical_accuracy的例子。y_true = 2; y_pred = (0.02, 0.05, 0.83, 0.1); acc = sparse_categorical_accuracy(y_true, y_pred)
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy')
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


def init_bert(model_path):
    dict_path = os.path.join(model_path, 'vocab.txt')
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    config_path = os.path.join(model_path, 'bert_config.json')
    checkpoint_path = os.path.join(model_path, 'bert_model.ckpt')
    # 加载预训练模型
    model = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='roberta',
        with_mlm=True
    )
    y_in = keras.layers.Input(shape=(None,))
    outputs = CrossEntropy(1)([y_in, model.output])
    train_model = keras.models.Model(model.inputs + [y_in], outputs)
    train_model.compile(optimizer=Adam(1e-5))

    return model, train_model, tokenizer


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self, model, valid_generator):
        super().__init__()
        self.best_val_acc = 0.
        self.model = model
        self.valid_generator = valid_generator

    def on_epoch_end(self, epoch, logs=None):
        # val_acc = evaluate(self.valid_generator, self.model)
        # if val_acc >= self.best_val_acc:
        #     self.best_val_acc = val_acc
        #     # self.model.save_weights(self.model_weights_f_path)
        # print(
        #     u'val_acc: %.5f, best_val_acc: %.5f\n' % (val_acc, self.best_val_acc)
        # )
        pass


class MergeType(Enum):
    CONCAT = 'concat'


class MlmBertEncoder(BaseEncoder):

    @property
    def dim(self):
        each_dim = len(self.key_token_index)
        if self.merge == self.CONCAT:
            return each_dim * len(self.mask_indxes)
        elif self.merge == self.MEAN:
            return each_dim

    CONCAT = 'concat'
    MEAN = 'mean'

    def __init__(self, model_path, train_data, dev_data, prefix, mask_idxes, labels, batch_size, merge=CONCAT,
                 max_len=256):
        self.train_data = train_data
        self.dev_data = dev_data
        self.model, self.train_model, self.tokenizer = init_bert(model_path)
        self.key_tokens = set(''.join(labels.values()))
        self.key_token_index = self.tokenizer.tokens_to_ids(self.key_tokens)
        self.train_generator = data_generator(train_data, batch_size, self.tokenizer, 256, prefix, mask_idxes, labels)
        self.dev_generator = data_generator(dev_data, batch_size, self.tokenizer, 256, prefix, mask_idxes, labels)
        self.prefix = prefix
        self.mask_indxes = mask_idxes
        self.merge = merge
        self.max_len = max_len
        self.flag = True if not mask_idxes else False

    def train(self, n_epoch=1):
        evaluator = Evaluator(self.model, self.dev_data)

        self.train_model.fit_generator(
            self.train_generator.forfit(),
            verbose=0,
            steps_per_epoch=len(self.train_generator),
            epochs=n_epoch,
            callbacks=[evaluator]
        )

    def get_prob(self, text, mask_ind_list):
        text = text[-self.max_len + 2:]
        token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.max_len)
        token_ids = token_ids[1:-1]
        segment_ids = segment_ids[1:-1]
        if self.flag:
            self.mask_indxes = [index for index, t_ids in enumerate(token_ids) if t_ids == 7233]
            mask_ind_list = self.mask_indxes
        for mask_ind in mask_ind_list:
            token_ids[mask_ind] = self.tokenizer._token_mask_id
        token_ids, segment_ids = to_array([token_ids], [segment_ids])

        # 用mlm模型预测被mask掉的部分
        emb = self.model.predict([token_ids, segment_ids])[0]
        matrix = []
        for ind in self.mask_indxes:
            token_emb = [emb[ind][key_ind] for key_ind in self.key_token_index]
            matrix.append(token_emb)

        if self.merge == self.CONCAT:
            return np.concatenate(matrix)

        elif self.merge == self.MEAN:
            return np.mean(matrix, axis=0)

    def encode(self, text):
        text = self.prefix + text
        vec = self.get_prob(text, self.mask_indxes)
        norm = np.apply_along_axis(np.linalg.norm, 0, vec)
        vec = vec / norm
        return vec
