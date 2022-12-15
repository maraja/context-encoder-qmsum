from scipy.special import softmax

# from transformers import BertTokenizer, TFBertMainLayer, TFBertForNextSentencePrediction, TFBertPreTrainedModel
# from transformers.modeling_tf_utils import TFPreTrainedModel, get_initializer, keras_serializable, shape_list
import tensorflow as tf
import numpy as np
import pickle
import math
import pickle
from transformers import DistilBertTokenizer
from .nlppreprocessing import (
    augment_sentences_and_labels,
    shuffle_sentences_and_labels,
    split_to_segments,
)
from .utils import flatten
from .dataset_params import DatasetParams

import sys

sys.path.append("../../")

from db.db import DB, AugmentedDB

from cached_property import cached_property

MODEL = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL)


class DistilBERTDataset(DatasetParams):
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
        max_segment_length=5
    ):
        processed_text_sentences = None
        processed_sentences = None
        processed_labels = None
        self.db = DB(dataset_type)
        self.augmented_db = AugmentedDB(dataset_type)
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
        regular_segments = self.db.get_random_segments_pct(
            pct_data=self.pct_data, max_segment_size=self.max_segment_length
        )
        augmented_segments = self.augmented_db.get_random_segments_pct(
            pct_data=self.augment_pct, max_segment_size=self.max_segment_length
        )

        return regular_segments + augmented_segments

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

    def _remove_duplicates(self, sentences, tokenized_sentences, labels):
        new_text_sentences = []
        new_sentences = []
        new_labels = []
        prev_sent = ""
        for t, s, l in zip(sentences, tokenized_sentences["input_ids"].numpy(), labels):
            if t == prev_sent:
                continue
            else:
                new_text_sentences.append(t)
                new_sentences.append(s)
                new_labels.append(l)
            prev_sent = t

        sentences = new_text_sentences
        tokenized_sentences = {"input_ids": tf.convert_to_tensor(new_sentences)}
        labels = new_labels
        return sentences, tokenized_sentences, labels

    def format_sentences_tri_input(self, sentences, pct_data=1):
        sentences = np.array(
            sentences["input_ids"][
                : (math.floor(pct_data * len(sentences["input_ids"])))
            ]
        )

        left_input = tf.convert_to_tensor([sentences[-1], *sentences[:-1]])

        mid_input = tf.convert_to_tensor(sentences)

        right_input = tf.convert_to_tensor([*sentences[1:], sentences[0]])

        return left_input, mid_input, right_input

    def format_sentences_tri_input_plus(self, sentences, pct_data=1):
        sentences = {
            "input_ids": np.array(
                sentences["input_ids"][
                    : (math.floor(pct_data * len(sentences["input_ids"])))
                ]
            ),
            #     'token_type_ids': np.array(sentences['token_type_ids']),
            "attention_mask": np.array(
                sentences["attention_mask"][
                    : (math.floor(pct_data * len(sentences["attention_mask"])))
                ]
            ),
        }

        left_input = {
            "input_ids": tf.convert_to_tensor(
                [sentences["input_ids"][-1], *sentences["input_ids"][:-1]]
            ),
            #     'token_type_ids': tf.convert_to_tensor([sentences['token_type_ids'][-1], *sentences['token_type_ids'][:-1]]),
            "attention_mask": tf.convert_to_tensor(
                [sentences["attention_mask"][-1], *sentences["attention_mask"][:-1]]
            ),
        }

        mid_input = {
            "input_ids": tf.convert_to_tensor(sentences["input_ids"]),
            #     'token_type_ids': tf.convert_to_tensor(sentences['token_type_ids']),
            "attention_mask": tf.convert_to_tensor(sentences["attention_mask"]),
        }

        right_input = {
            "input_ids": tf.convert_to_tensor(
                [*sentences["input_ids"][1:], sentences["input_ids"][0]]
            ),
            #     'token_type_ids': tf.convert_to_tensor([*sentences['token_type_ids'][1:], sentences['token_type_ids'][0]]),
            "attention_mask": tf.convert_to_tensor(
                [*sentences["attention_mask"][1:], sentences["attention_mask"][0]]
            ),
        }

        return left_input, mid_input, right_input

    def format_sentences_5_input(self, sentences, pct_data=1):
        sentences = np.array(
            sentences["input_ids"][
                : (math.floor(pct_data * len(sentences["input_ids"])))
            ]
        )

        left_1_input = tf.convert_to_tensor(
            [sentences[-2], sentences[-1], *sentences[:-2]]
        )

        left_2_input = tf.convert_to_tensor([sentences[-1], *sentences[:-1]])

        mid_input = tf.convert_to_tensor(sentences)

        right_1_input = tf.convert_to_tensor([*sentences[1:], sentences[0]])

        right_2_input = tf.convert_to_tensor(
            [*sentences[2:], sentences[0], sentences[1]]
        )

        return left_1_input, left_2_input, mid_input, right_1_input, right_2_input

    def process(self):
        sentences, labels = self.sentences, self.labels

        tokenized_sentences = tokenizer.batch_encode_plus(
            sentences,
            max_length=self.max_seq_length,
            truncation=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            add_special_tokens=True,
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
