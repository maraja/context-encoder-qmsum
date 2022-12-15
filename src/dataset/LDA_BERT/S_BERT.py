from .model import Topic_Model
from .utils import *
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import math

import warnings

warnings.filterwarnings("ignore", category=Warning)

import argparse


class S_BERT:
    """
    S_BERT class to provide vectorized sentences with LDA.
    This is a supervised learning approach and requires the dataset to be
    handed over before initializing.
    """

    def __init__(self, sentences, topics, token_lists, pct_data=1):
        self.sentences = sentences
        self.topics = topics
        self.pct_data = pct_data
        self.sentences = self.sentences[: math.floor(len(self.sentences) * pct_data)]
        self.token_lists = token_lists
        self.model = None

    def _compile(self, method):
        # Define the topic model object
        self.model = Topic_Model(k=self.topics, method=method)
        # # Fit the topic model by chosen method
        # self.model.fit(self.sentences, self.token_lists)

    def vectorize(self, method="S_BERT"):
        if not self.model:
            self._compile(method)
        vectors = self.model.vectorize(self.sentences, self.token_lists, method=method)
        return vectors
