import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Lambda,
    Input,
    Attention,
    Dropout,
    Embedding,
    Dense,
    Flatten,
    Concatenate,
    Conv2D,
    MaxPooling2D,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    GRU,
    Softmax,
)
from transformers import (
    TFBertMainLayer,
    AutoConfig,
    TFDistilBertMainLayer,
    DistilBertConfig,
    AlbertConfig,
    TFAlbertMainLayer,
)

MODEL_DISTILBERT_CASED = "distilbert-base-cased"
MODEL_DISTILBERT_UNCASED = "distilbert-base-uncased"
MODEL_ALBERT = "albert-base-v2"

supported_bert_types = [MODEL_DISTILBERT_CASED, MODEL_DISTILBERT_UNCASED, MODEL_ALBERT]
# distilbert_model = TFDistilBertModel.from_pretrained(MODEL)


def pool_output(input_tensor):
    bert_full_output = tf.transpose(input_tensor, [0, 2, 1])
    bert_pooled_output = tf.reduce_mean(bert_full_output, 2)
    return bert_pooled_output


def stack_input(input_tensor, cols: int, rows: int):
    """This function will create a 2D 8x96 matrix for each input vector, given a 768 vector

    Args:
        input_tensor (list): the input tensor.

    Returns:
        arr: 2d matrix of 8x96
    """
    # assert (len(input_tensor)/cols) == rows
    return tf.map_fn(fn=lambda t: tf.reshape(t, [rows, cols]), elems=input_tensor)


class ContextEncoderMixin:
    def init_params(self, params, kwargs):
        for param in params:
            self.__dict__[param] = kwargs[param] if param in kwargs else False

        self.bert_type = (
            kwargs["bert_type"] if "bert_type" in kwargs else "distilbert-base-cased"
        )

    def init_bert_stack(self, bert_type):
        if bert_type not in supported_bert_types:
            raise Exception("provided bert type not supported")
        if self.bert_type.startswith("distilbert"):
            self.bert_config = DistilBertConfig.from_pretrained(bert_type)
            self.bert = TFDistilBertMainLayer(self.bert_config, name="distilbert")
        elif self.bert_type.startswith("albert"):
            self.bert_config = AlbertConfig.from_pretrained(bert_type)
            self.bert = TFAlbertMainLayer(self.bert_config, name="albert")
        else:
            self.bert_config = AutoConfig.from_pretrained(bert_type)
            self.bert = TFBertMainLayer(self.bert_config, name="bert")  # Embeddings
        self.bert.trainable = self.bert_trainable

    def init_dense_stack(self):
        self.dense_input_1 = Dense(
            self.dense_neurons, activation="tanh", name="dense_input_left"
        )
        self.dense_input_2 = Dense(
            self.dense_neurons, activation="tanh", name="dense_input_mid"
        )
        self.dense_input_3 = Dense(
            self.dense_neurons, activation="tanh", name="dense_input_right"
        )

    def init_lstm_stack(self):
        if self.lstm_dropout_percentage and self.lstm_size:
            self.bi_lstm_1 = Bidirectional(
                LSTM(
                    self.lstm_size,
                    dropout=self.lstm_dropout_percentage,
                    return_sequences=True,
                    return_state=True,
                    name="lstm_left",
                ),
                name="bi-directional_left",
            )
            self.bi_lstm_2 = Bidirectional(
                LSTM(
                    self.lstm_size,
                    dropout=self.lstm_dropout_percentage,
                    return_sequences=True,
                    return_state=True,
                    name="lstm_mid",
                ),
                name="bi-directional_mid",
            )
            self.bi_lstm_3 = Bidirectional(
                LSTM(
                    self.lstm_size,
                    dropout=self.lstm_dropout_percentage,
                    return_sequences=True,
                    return_state=True,
                    name="lstm_right",
                ),
                name="bi-directional_right",
            )
        elif self.lstm_size:
            self.bi_lstm_1 = Bidirectional(
                LSTM(
                    self.lstm_size,
                    return_sequences=True,
                    return_state=True,
                    name="lstm_left",
                ),
                name="bi-directional_left",
            )
            self.bi_lstm_2 = Bidirectional(
                LSTM(
                    self.lstm_size,
                    return_sequences=True,
                    return_state=True,
                    name="lstm_mid",
                ),
                name="bi-directional_mid",
            )
            self.bi_lstm_3 = Bidirectional(
                LSTM(
                    self.lstm_size,
                    return_sequences=True,
                    return_state=True,
                    name="lstm_right",
                ),
                name="bi-directional_right",
            )

    def init_cnn_stack(self):
        self.conv1_1 = Conv1D(
            self.cnn_filters,
            self.cnn_kernel_size,
            activation="relu",
            name="conv1D_left",
        )
        self.conv1_2 = Conv1D(
            self.cnn_filters, self.cnn_kernel_size, activation="relu", name="conv1D_mid"
        )
        self.conv1_3 = Conv1D(
            self.cnn_filters,
            self.cnn_kernel_size,
            activation="relu",
            name="conv1D_right",
        )

    def init_max_pool_stack(self):
        self.pool_1 = MaxPooling1D(pool_size=self.pool_size, name="maxpooling1D_left")
        self.pool_2 = MaxPooling1D(pool_size=self.pool_size, name="maxpooling1D_mid")
        self.pool_3 = MaxPooling1D(pool_size=self.pool_size, name="maxpooling1D_right")

    def init_attention_stack(self):
        self.left_attention = Attention(name="left_context")
        self.mid_attention = Attention(name="mid_sentence")
        self.right_attention = Attention(name="right_context")

    def init_output_stack(self):
        self.dense_output = None
        if self.output_bias:
            self.dense_output = Dense(
                1,
                activation="sigmoid",
                name="dense_output",
                bias_initializer=tf.keras.initializers.Constant(self.output_bias),
            )
        else:
            self.dense_output = Dense(1, activation="sigmoid", name="dense_output")

    def init_dropout_stack(self):
        if self.final_dropout:
            self.dropout = Dropout(self.final_dropout, name="final_dropout")
        else:
            self.dropout = None

    def log_model(self, checkpoint_filepath, **kwargs):
        checkpoint_filepath = checkpoint_filepath.replace("/checkpoint", "")
        from pathlib import Path

        Path(checkpoint_filepath).mkdir(parents=True, exist_ok=True)
        handle = open("{}/model.txt".format(checkpoint_filepath), "w+")
        for key, value in kwargs.items():
            handle.write("{}: {}\n".format(key, value))
        handle.write(
            "{}: {}\n".format(
                "lstm_dropout",
                self.lstm_dropout_percentage if self.lstm_dropout_percentage else False,
            )
        )
        handle.write(
            "{}: {}\n".format("lstm_size", self.lstm_size if self.lstm_size else False)
        )
        handle.write(
            "{}: {}\n".format(
                "final_dropout", self.final_dropout if self.final_dropout else False
            )
        )
        handle.write(
            "{}: {}\n".format(
                "dense_neurons", self.dense_neurons if self.dense_neurons else False
            )
        )
        handle.write(
            "{}: {}\n".format("output_bias", True if self.output_bias else False)
        )
        handle.write(
            "{}: {}\n".format("bert_finetuned", True if self.bert_trainable else False)
        )
        handle.close()
