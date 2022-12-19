# from transformers import BertTokenizer, TFBertMainLayer, TFBertForNextSentencePrediction, TFBertPreTrainedModel
# from transformers.modeling_tf_utils import TFPreTrainedModel, get_initializer, keras_serializable, shape_list
from transformers import AutoTokenizer, TFAutoModel, AutoModel
from cached_property import cached_property
from db.db import DB, AugmentedDB
from src.dataset.LDA_BERT.LDA_BERT import LDA
import tensorflow as tf
import numpy as np
import pickle
import math
import config
import pathlib
from .dataset_params import DatasetParams

from .nlppreprocessing import (
    preprocess_sent,
    preprocess_sent_simple,
    preprocess_word,
)
from .utils import flatten
import sys
import os

sys.path.insert(0, config.root_path)


sbert_model = "sentence-transformers/bert-base-nli-stsb-mean-tokens"
tokenizer = AutoTokenizer.from_pretrained(sbert_model)


class LDABERT3Dataset(DatasetParams):
    def __init__(
        self,
        *,
        dataset_slice="training",
        dataset_type="default",
        pct_data=0.005,
        max_seq_length=256,
        random=False,
        augment_pct=0.0,
        remove_duplicates=False,
        max_segment_length=5,
        lda_topics=12,
        lda_gamma=15,
        split="train",
        artificial_segments=False
    ):
        self.topics = lda_topics
        self.lda_gamma = lda_gamma
        self.db = DB(dataset_type)
        self.augmented_db = AugmentedDB(dataset_type)
        self.split = split
        self.artificial_segments = artificial_segments
        super().__init__(
            dataset_slice=dataset_slice,
            dataset_type=dataset_type,
            pct_data=pct_data,
            max_seq_length=max_seq_length,
            random=random,
            augment_pct=augment_pct,
            remove_duplicates=remove_duplicates,
            max_segment_length=max_segment_length,
        )

    @cached_property
    def data_segments(self):
        if self.split == "train":
            regular_segments = self.db.get_random_segments_pct(
                pct_data=self.pct_data,
                split=self.split,
                max_segment_size=self.max_segment_length,
                artificial_segments=self.artificial_segments,
            )
            augmented_segments = self.augmented_db.get_random_segments_pct(
                pct_data=self.augment_pct,
                max_segment_size=self.max_segment_length,
                artificial_segments=self.artificial_segments,
            )

            return regular_segments + augmented_segments

        # testing doesn't need augmented data
        return self.db.get_random_segments_pct(
            pct_data=self.pct_data,
            split=self.split,
            max_segment_size=self.max_segment_length,
        )

    @cached_property
    def data(self):
        return flatten(self.data_segments)

    @cached_property
    def num_samples(self):
        return len(self.data)

    @cached_property
    def sentences(self):
        return [x[1] for x in self.data]

    @cached_property
    def labels(self):
        return [x[2] for x in self.data]

    @cached_property
    def sentence_segments(self):
        return [[y[1] for y in x] for x in self.data_segments]

    @cached_property
    def label_segments(self):
        return [[y[2] for y in x] for x in self.data_segments]

    def _remove_duplicates(self, sentences, labels):
        new_sentences = []
        new_labels = []
        prev_sent = ""
        for t, l in zip(sentences, labels):
            if t == prev_sent:
                continue
            else:
                new_sentences.append(t)
                new_labels.append(l)
            prev_sent = t

        return new_sentences, new_labels

    def preprocess_lda(self, sentences, labels):
        """
        Preprocess the data
        """

        print("Preprocessing raw texts ...")
        new_sentences = []  # sentence level preprocessed
        token_lists = []  # word level preprocessed
        # idx_in = []  # index of sample selected
        new_labels = []
        #     samp = list(range(100))
        print("sentences length", len(sentences))
        for i, sent in enumerate(sentences):
            sentence = preprocess_sent_simple(sent)
            token_list = preprocess_word(sentence)
            # if token_list:
            # idx_in.append(idx)
            new_sentences.append(sentence)
            token_lists.append(token_list)
            new_labels.append(labels[i])
            print(
                "{} %".format(
                    str(np.round((i + 1) / len(sentences) * 100, 2))),
                end="\r",
            )
        print("Preprocessing raw texts. Done!")
        self.lda_sentences, self.lda_token_lists, self.lda_new_labels = (
            new_sentences,
            token_lists,
            new_labels,
        )
        print("lda sentences length", len(self.lda_sentences))
        return new_sentences, token_lists, new_labels

    def format_sentences_tri_input(self, sentences):

        left_input = tf.convert_to_tensor([sentences[-1], *sentences[:-1]])

        mid_input = tf.convert_to_tensor(sentences)

        right_input = tf.convert_to_tensor([*sentences[1:], sentences[0]])

        return left_input, mid_input, right_input

    def format_sentences_tri_input_plus(self, sentences):
        left_input = {
            "input_ids": tf.convert_to_tensor(
                [sentences["input_ids"][-1], *sentences["input_ids"][:-1]]
            ),
            "token_type_ids": tf.convert_to_tensor(
                [sentences["token_type_ids"][-1], *
                    sentences["token_type_ids"][:-1]]
            ),
            "attention_mask": tf.convert_to_tensor(
                [sentences["attention_mask"][-1], *
                    sentences["attention_mask"][:-1]]
            ),
        }

        mid_input = {
            "input_ids": tf.convert_to_tensor(sentences["input_ids"]),
            "token_type_ids": tf.convert_to_tensor(sentences["token_type_ids"]),
            "attention_mask": tf.convert_to_tensor(sentences["attention_mask"]),
        }

        right_input = {
            "input_ids": tf.convert_to_tensor(
                [*sentences["input_ids"][1:], sentences["input_ids"][0]]
            ),
            "token_type_ids": tf.convert_to_tensor(
                [*sentences["token_type_ids"][1:], sentences["token_type_ids"][0]]
            ),
            "attention_mask": tf.convert_to_tensor(
                [*sentences["attention_mask"][1:], sentences["attention_mask"][0]]
            ),
        }

        return left_input, mid_input, right_input

    def generate_vectors(self):
        self.lda = LDA(self.lda_sentences, self.topics, self.lda_token_lists)
        self.lda_vectors = self.lda.vectorize(method="LDA")
        return self.lda_vectors

    def create_vectors(self, dataset_split, dataset_type, filename):
        print("root path", config.root_path)
        self.preprocess_lda(self.sentences, self.labels)
        vectors = self.generate_vectors()
        absolute_filepath = os.path.join(
            config.root_path,
            "data",
            "lda_bert_2",
            "generated_vectors",
            dataset_split,
            dataset_type,
            filename,
        )
        pickle.dump(
            [vectors, self.labels, self.sentences, self.processed_sentences],
            open(absolute_filepath, "wb"),
        )
        print("saving vectors...", len(vectors),
              len(self.labels), len(self.sentences))
        return vectors, self.labels, self.sentences, self.processed_sentences

    def get_saved_vectors(self, dataset_split, dataset_type, filename):
        absolute_filepath = os.path.join(
            config.root_path,
            "data",
            "lda_bert_2",
            "generated_vectors",
            dataset_split,
            dataset_type,
            filename,
        )
        try:
            vectors = pickle.load(open(absolute_filepath, "rb", buffering=0))
            return (
                vectors[0] if len(vectors[0]) else [],
                vectors[1] if len(vectors[1]) else [],
                vectors[2] if len(vectors[2]) else [],
                vectors[3] if len(vectors[3]) else [],
            )
        except Exception as e:
            print("something went wrong", e)
            return [], [], [], []

    def process(self):
        sentence_segments, label_segments = self.sentence_segments, self.label_segments
        sentences, labels = self.sentences, self.labels

        tokenized_sentences = tokenizer(
            sentences,
            padding=True,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="tf",
        )

        # convert to a vertical stacked array.
        labels = np.expand_dims(np.array(labels), axis=1)

        if self.remove_duplicates:
            sentences, tokenized_sentences, labels = self._remove_duplicates(
                sentences, tokenized_sentences, labels
            )

        self.processed_text_sentences = sentences
        self.processed_sentences = tokenized_sentences
        self.processed_labels = labels
        return sentences, tokenized_sentences, labels
