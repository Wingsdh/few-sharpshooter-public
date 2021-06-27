# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    vector_classifier.py
   Description :
   Author :       Wings DH
   Time：         2021/5/14 下午10:04
-------------------------------------------------
   Change Activity:
                   2021/5/14: Create
-------------------------------------------------
"""
from collections import defaultdict
from typing import List
import numpy as np

from annoy import AnnoyIndex

from modeling.base_encoder import BaseEncoder
from modeling.classifier import LabelData
from modeling.classifier import BaseClassifier


class SentenceRetriever(object):
    @staticmethod
    def build_annoy_index(all_vec, n_dim):
        annoy_index = AnnoyIndex(n_dim, 'dot')
        for idx, vec in enumerate(all_vec):
            annoy_index.add_item(idx, vec)
        annoy_index.build(10)
        return annoy_index

    def __init__(self, encoder: BaseEncoder, matrix, data, dim=None):
        self.encoder = encoder
        self.data = data
        self.text_2_position = {t.text: i for i, t in enumerate(self.data)}
        self.matrix = matrix
        if dim is None:
            dim = self.encoder.dim

        self.annoy_index = self.build_annoy_index(self.matrix, dim)

    def retrieve(self, text=None, vec=None, n_top=9):
        # 向量检索
        if vec is None:
            vec = self.encoder.encode(text)

        rst = self.annoy_index.get_nns_by_vector(vec, n_top, include_distances=True)
        most_sim_texts = [self.data[ind] for ind in rst[0]]
        scores = [s for s in rst[1]]
        return most_sim_texts, scores


class RetrieverClassifier(BaseClassifier):
    def __init__(self, encoder: BaseEncoder, train_data: List[LabelData], n_top=7):
        # 加载分类器
        matrix = [encoder.encode(t.text) for t in train_data]
        self.retriever = SentenceRetriever(encoder, matrix, train_data)
        self.n_top = n_top

    def classify(self, text):
        most_sim_datas, scores = self.retriever.retrieve(text, n_top=self.n_top)
        label_2_count = defaultdict(list)
        for d, s in zip(most_sim_datas, scores):
            label_2_count[d.label].append(s)

        label = max(label_2_count.keys(), key=lambda x: np.sum(label_2_count[x]))
        return label, most_sim_datas


class PairSentenceRetriever(object):
    @staticmethod
    def build_annoy_index(all_vec, n_dim):
        annoy_index = AnnoyIndex(n_dim, 'dot')
        for idx, vec in enumerate(all_vec):
            annoy_index.add_item(idx, vec)
        annoy_index.build(10)
        return annoy_index

    def __init__(self, encoder, matrix, data, dim=None):
        self.encoder = encoder
        self.data = data
        self.matrix = matrix
        if dim is None:
            dim = self.encoder.dim

        self.annoy_index = self.build_annoy_index(self.matrix, dim)

    def retrieve(self, text0, text1, vec=None, n_top=9):
        # 向量检索
        if vec is None:
            vec = self.encoder.encode(text0, text1)

        rst = self.annoy_index.get_nns_by_vector(vec, n_top, include_distances=True)
        most_sim_texts = [self.data[ind] for ind in rst[0]]
        scores = [s for s in rst[1]]
        return most_sim_texts, scores


class PairRetrieverClassifier(object):
    def __init__(self, encoder, train_data, n_top=7):
        # 加载分类器
        matrix = [encoder.encode(t.text0, t.text1) for t in train_data]
        self.retriever = PairSentenceRetriever(encoder, matrix, train_data)
        self.n_top = n_top

    def classify(self, text0, text1):
        most_sim_datas, scores = self.retriever.retrieve(text0, text1, n_top=self.n_top)
        label_2_count = defaultdict(int)
        for d, s in zip(most_sim_datas, scores):
            label_2_count[d.label] += s

        label = max(label_2_count.keys(), key=lambda x: label_2_count[x])
        print(label_2_count[label])
        return label


class MatrixClassifier(BaseClassifier):
    def __init__(self, encoder: BaseEncoder, train_data: List[LabelData]):
        self.encoder = encoder
        # 加载分类器
        self.matrix = []
        for d in train_data:
            emb = self.encoder.encode(d.text)
            self.matrix.append(emb)
        self.data = train_data

        union_matrix = []
        for emb in self.matrix:
            union_emb = np.einsum('i,ji->j', emb, self.matrix)
            union_emb = np.concatenate([emb, union_emb])
            union_matrix.append(union_emb)

        self.matrix = np.array(self.matrix, dtype=np.int)
        self.retriever = SentenceRetriever(encoder, union_matrix, train_data, self.encoder.dim + 32)

    def classify(self, text):
        emb = self.encoder.encode(text)
        union_emb = np.einsum('i,ji->j', emb, self.matrix)
        union_emb = np.concatenate([emb, union_emb])
        most_sim_datas, scores = self.retriever.retrieve(text, vec=union_emb)
        label_2_count = defaultdict(int)
        for d, s in zip(most_sim_datas, scores):
            label_2_count[d.label] += s

        label = max(label_2_count.keys(), key=lambda x: label_2_count[x])
        return label
