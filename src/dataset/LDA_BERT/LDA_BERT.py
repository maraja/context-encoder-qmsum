import argparse
from .model import Topic_Model
from .utils import *
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import math

import warnings

warnings.filterwarnings("ignore", category=Warning)


class LDA_BERT:
    """
    LDA_BERT class to provide vectorized sentences with LDA.
    This is a supervised learning approach and requires the dataset to be
    handed over before initializing.
    """

    def __init__(self, sentences, topics, token_lists, lda_gamma=15, pct_data=1):
        self.sentences = sentences
        self.topics = topics
        self.pct_data = pct_data
        self.lda_gamma = lda_gamma
        self.sentences = self.sentences[: math.floor(
            len(self.sentences) * pct_data)]
        self.token_lists = token_lists
        self.model = None

    def _compile(self, method):
        # Define the topic model object
        self.model = Topic_Model(
            k=self.topics, lda_gamma=self.lda_gamma, method=method)
        # # Fit the topic model by chosen method
        # self.model.fit(self.sentences, self.token_lists)

    def vectorize(self, method="LDA_BERT"):
        if not self.model:
            self._compile(method)
        return self.model.vectorize(self.sentences, self.token_lists, method=method)


class LDA:
    def __init__(self, sentences, topics, token_lists, lda_gamma=15, pct_data=1):
        self.sentences = sentences
        self.topics = topics
        self.lda_gamma = lda_gamma
        self.token_lists = token_lists
        self.model = None

    def _compile(self, method):
        # Define the topic model object
        self.model = Topic_Model(
            k=self.topics, lda_gamma=self.lda_gamma, method=method)
        # # Fit the topic model by chosen method
        # self.model.fit(self.sentences, self.token_lists)

    def vectorize(self, method="LDA"):
        if not self.model:
            self._compile(method)
        return self.model.vectorize(self.sentences, self.token_lists, method=method)
