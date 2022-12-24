from re import S
import tensorflow as tf

from tensorflow.python.keras.layers import Concatenate, Lambda
from transformers import AutoTokenizer, AutoModel, TFAutoModel

from .context_encoder import ContextEncoderMixin, stack_input

SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/bert-base-nli-mean-tokens"
# SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/distilbert-base-nli-mean-tokens"

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    unsqueezed = tf.expand_dims(attention_mask, -1)
    expanded = tf.broadcast_to(unsqueezed, tf.shape(token_embeddings))
    input_mask_expanded = tf.cast(expanded, tf.float32)
    summed = tf.reduce_mean(tf.math.multiply(token_embeddings, input_mask_expanded), 1)
    clamped = tf.clip_by_value(
        tf.reduce_sum(input_mask_expanded, 1), clip_value_min=1e-9, clip_value_max=1e9
    )
    return tf.math.divide(summed, clamped)


class ContextEncoderSimple(tf.keras.Model, ContextEncoderMixin):
    def __init__(self, *inputs, **kwargs):
        super().__init__()
        self.output_bias = None

        # self.tokenizer = AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMERS_MODEL)
        self.sbert = TFAutoModel.from_pretrained(
            SENTENCE_TRANSFORMERS_MODEL, from_pt=True
        )

        self.init_params(
            [
                "final_dropout",
                "dense_neurons",
                "output_bias",
                "max_sentence_length",
                "bert_trainable",
                "gamma",
            ],
            kwargs,
        )

        self.sbert.trainable = self.bert_trainable

        self.init_dense_stack()

        self.init_output_stack()

        self.init_dropout_stack()

    def call(self, inputs, **kwargs):
        bert_inputs = inputs[0:3]
        lda_inputs = inputs[3:]

        # Compute token embeddings
        model_output_left = self.sbert(**bert_inputs[0])
        model_output_mid = self.sbert(**bert_inputs[1])
        model_output_right = self.sbert(**bert_inputs[2])

        # Perform pooling. In this case, max pooling.
        sentence_embeddings_left = mean_pooling(
            model_output_left, bert_inputs[0]["attention_mask"]
        )
        sentence_embeddings_mid = mean_pooling(
            model_output_mid, bert_inputs[1]["attention_mask"]
        )
        sentence_embeddings_right = mean_pooling(
            model_output_right, bert_inputs[2]["attention_mask"]
        )

        # lda_output = [[15] * 10] * sentence_embeddings_left.shape[0]

        lda_left_input = Lambda(lambda x: x * self.gamma)(lda_inputs[0])
        lda_mid_input = Lambda(lambda x: x * self.gamma)(lda_inputs[1])
        lda_right_input = Lambda(lambda x: x * self.gamma)(lda_inputs[2])

        se_lda_left = Concatenate(name="lda_left")(
            [sentence_embeddings_left, lda_left_input]
        )
        se_lda_mid = Concatenate(name="lda_mid")(
            [sentence_embeddings_mid, lda_mid_input]
        )
        se_lda_right = Concatenate(name="lda_right")(
            [sentence_embeddings_right, lda_right_input]
        )

        left_output = self.dense_input_1(se_lda_left)
        mid_output = self.dense_input_2(se_lda_mid)
        right_output = self.dense_input_3(se_lda_right)

        concatenated_layer = Concatenate(name="fully_connected")(
            [left_output, mid_output, right_output]
        )

        if self.dropout:
            concatenated_layer = self.dropout(concatenated_layer)

        dense = self.dense_output(concatenated_layer)

        return dense


class ContextEncoderConv(tf.keras.Model, ContextEncoderMixin):
    def __init__(self, *inputs, **kwargs):
        super().__init__()
        self.output_bias = None

        # self.tokenizer = AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMERS_MODEL)
        self.sbert = TFAutoModel.from_pretrained(
            SENTENCE_TRANSFORMERS_MODEL, from_pt=True
        )

        self.init_params(
            [
                "cnn_filters",
                "cnn_kernel_size",
                "pool_size",
                "final_dropout",
                "dense_neurons",
                "output_bias",
                "max_sentence_length",
                "bert_trainable",
                "gamma",
            ],
            kwargs,
        )

        self.sbert.trainable = self.bert_trainable

        self.init_cnn_stack()
        self.init_max_pool_stack()

        self.init_dense_stack()

        self.init_output_stack()

        self.init_dropout_stack()

    def call(self, inputs, **kwargs):
        bert_inputs = inputs[0:3]
        lda_inputs = inputs[3:]

        # Compute token embeddings
        model_output_left = self.sbert(**bert_inputs[0])
        model_output_mid = self.sbert(**bert_inputs[1])
        model_output_right = self.sbert(**bert_inputs[2])

        # Perform pooling. In this case, max pooling.
        sentence_embeddings_left = mean_pooling(
            model_output_left, bert_inputs[0]["attention_mask"]
        )  # 768
        sentence_embeddings_mid = mean_pooling(
            model_output_mid, bert_inputs[1]["attention_mask"]
        )  # 768
        sentence_embeddings_right = mean_pooling(
            model_output_right, bert_inputs[2]["attention_mask"]
        )  # 768

        # lda_output = [[15] * 10] * sentence_embeddings_left.shape[0]

        lda_left_input = Lambda(lambda x: x * self.gamma)(lda_inputs[0])  # 12
        lda_mid_input = Lambda(lambda x: x * self.gamma)(lda_inputs[1])  # 12
        lda_right_input = Lambda(lambda x: x * self.gamma)(lda_inputs[2])  # 12

        se_lda_left = Concatenate(name="lda_left")(
            [sentence_embeddings_left, lda_left_input]
        )  # 780
        se_lda_mid = Concatenate(name="lda_mid")(
            [sentence_embeddings_mid, lda_mid_input]
        )  # 780
        se_lda_right = Concatenate(name="lda_right")(
            [sentence_embeddings_right, lda_right_input]
        )  # 780

        stacked_left_input = stack_input(
            se_lda_left, int(se_lda_left.shape[1] / 10), 10
        )
        stacked_mid_input = stack_input(se_lda_mid, int(se_lda_mid.shape[1] / 10), 10)
        stacked_right_input = stack_input(
            se_lda_right, int(se_lda_right.shape[1] / 10), 10
        )

        conv1_1 = self.conv1_1(stacked_left_input)
        conv1_2 = self.conv1_2(stacked_mid_input)
        conv1_3 = self.conv1_3(stacked_right_input)

        pool_1 = self.pool_1(conv1_1)
        pool_2 = self.pool_2(conv1_2)
        pool_3 = self.pool_3(conv1_3)

        left_output = self.dense_input_1(pool_1)
        mid_output = self.dense_input_2(pool_2)
        right_output = self.dense_input_3(pool_3)

        concatenated_layer = Concatenate(name="fully_connected")(
            [left_output, mid_output, right_output]
        )

        if self.dropout:
            concatenated_layer = self.dropout(concatenated_layer)

        dense = self.dense_output(concatenated_layer)

        return dense


class ContextEncoderComplex(tf.keras.Model, ContextEncoderMixin):
    def __init__(self, *inputs, **kwargs):
        super().__init__()
        self.output_bias = None

        # self.tokenizer = AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMERS_MODEL)
        self.sbert = TFAutoModel.from_pretrained(
            SENTENCE_TRANSFORMERS_MODEL, from_pt=True
        )

        self.init_params(
            [
                "lstm_dropout_percentage",
                "lstm_size",
                "cnn_filters",
                "cnn_kernel_size",
                "pool_size",
                "final_dropout",
                "dense_neurons",
                "output_bias",
                "max_sentence_length",
                "bert_trainable",
                "gamma",
            ],
            kwargs,
        )

        self.sbert.trainable = self.bert_trainable

        self.init_lstm_stack()
        self.init_attention_stack()
        self.init_cnn_stack()
        self.init_max_pool_stack()

        self.init_dense_stack()

        self.init_output_stack()

        self.init_dropout_stack()

    def call(self, inputs, **kwargs):
        bert_inputs = inputs[0:3]
        lda_inputs = inputs[3:]

        # Compute token embeddings
        model_output_left = self.sbert(**bert_inputs[0])
        model_output_mid = self.sbert(**bert_inputs[1])
        model_output_right = self.sbert(**bert_inputs[2])

        print("sbert_output", model_output_left)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings_left = mean_pooling(
            model_output_left, bert_inputs[0]["attention_mask"]
        )  # 768
        sentence_embeddings_mid = mean_pooling(
            model_output_mid, bert_inputs[1]["attention_mask"]
        )  # 768
        sentence_embeddings_right = mean_pooling(
            model_output_right, bert_inputs[2]["attention_mask"]
        )  # 768

        # lda_output = [[15] * 10] * sentence_embeddings_left.shape[0]

        lda_left_input = Lambda(lambda x: x * self.gamma)(lda_inputs[0])  # 12
        lda_mid_input = Lambda(lambda x: x * self.gamma)(lda_inputs[1])  # 12
        lda_right_input = Lambda(lambda x: x * self.gamma)(lda_inputs[2])  # 12

        # print("lda_left_input", lda_left_input)

        se_lda_left = Concatenate(name="lda_left")(
            [sentence_embeddings_left, lda_left_input]
        )  # 780
        se_lda_mid = Concatenate(name="lda_mid")(
            [sentence_embeddings_mid, lda_mid_input]
        )  # 780
        se_lda_right = Concatenate(name="lda_right")(
            [sentence_embeddings_right, lda_right_input]
        )  # 780

        stacked_left_input = stack_input(
            se_lda_left, int(se_lda_left.shape[1] / 10), 10
        )
        stacked_mid_input = stack_input(se_lda_mid, int(se_lda_mid.shape[1] / 10), 10)
        stacked_right_input = stack_input(
            se_lda_right, int(se_lda_right.shape[1] / 10), 10
        )

        conv1_1 = self.conv1_1(stacked_left_input)
        conv1_2 = self.conv1_2(stacked_mid_input)
        conv1_3 = self.conv1_3(stacked_right_input)

        pool_1 = self.pool_1(conv1_1)
        pool_2 = self.pool_2(conv1_2)
        pool_3 = self.pool_3(conv1_3)

        # https://stackoverflow.com/questions/49313650/how-could-i-get-both-the-final-hidden-state-and-sequence-in-a-lstm-layer-when-us
        lstm_1, forward_1_h, forward_1_c, backward_1_h, backward_1_c = self.bi_lstm_1(
            pool_1
        )
        lstm_2, forward_2_h, forward_2_c, backward_2_h, backward_2_c = self.bi_lstm_2(
            pool_2
        )
        lstm_3, forward_3_h, forward_3_c, backward_3_h, backward_3_c = self.bi_lstm_3(
            pool_3
        )

        state_1_h = Concatenate()([forward_1_h, backward_1_h])
        state_1_c = Concatenate()([forward_1_c, backward_1_c])
        state_2_h = Concatenate()([forward_2_h, backward_2_h])
        state_2_c = Concatenate()([forward_2_c, backward_2_c])
        state_3_h = Concatenate()([forward_3_h, backward_3_h])
        state_3_c = Concatenate()([forward_3_c, backward_3_c])
        
        # print("state_1_h", state_1_h)
        # print("state_1_c", state_1_c)

        # attention layer takes query and values.
        # In an LSTM, the query is the hidden (cell state) and values is output (hidden state)
        left_context = self.left_attention([state_1_c, state_1_h])
        mid_sentence = self.mid_attention([state_2_c, state_2_h])
        right_context = self.right_attention([state_3_c, state_3_h])
        # print("left_context", left_context)

        left_output = self.dense_input_1(se_lda_left)
        mid_output = self.dense_input_2(se_lda_mid)
        right_output = self.dense_input_3(se_lda_right)

        concatenated_layer = Concatenate(name="fully_connected")(
            [left_context, mid_sentence, right_context]
        )
        # print("concatenated_layer", concatenated_layer)

        if self.dropout:
            concatenated_layer = self.dropout(concatenated_layer)
            # print("concatenated_layer_dropout", concatenated_layer)

        dense = self.dense_output(concatenated_layer)
        # print("dense", dense)

        return dense
