from re import S
import tensorflow as tf

from tensorflow.python.keras.layers import Concatenate

from .context_encoder import ContextEncoderMixin, stack_input


class ContextEncoder(tf.keras.Model, ContextEncoderMixin):
    def __init__(self, *inputs, **kwargs):
        super().__init__()
        self.output_bias = None

        self.init_params(
            ["final_dropout", "dense_neurons", "output_bias"], kwargs,
        )

        self.init_dense_stack()

        self.init_output_stack()

        self.init_dropout_stack()

    def call(self, inputs, **kwargs):
        left_input = inputs[0]
        mid_input = inputs[1]
        right_input = inputs[2]

        left_input = self.dense_input_1(left_input)
        mid_input = self.dense_input_2(mid_input)
        right_input = self.dense_input_3(right_input)

        concatenated_layer = Concatenate(name="fully_connected")(
            [left_input, mid_input, right_input]
        )

        if self.dropout:
            concatenated_layer = self.dropout(concatenated_layer)

        dense = self.dense_output(concatenated_layer)

        return dense


class ContextEncoderComplex(tf.keras.Model, ContextEncoderMixin):
    def __init__(self, *inputs, **kwargs):
        super().__init__()
        self.output_bias = None

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
                "bert_trainable",
                "bert_type",
            ],
            kwargs,
        )

        self.init_lstm_stack()
        self.init_attention_stack()
        self.init_cnn_stack()
        self.init_max_pool_stack()

        self.init_dense_stack()

        self.init_output_stack()

        self.init_dropout_stack()

    def call(self, inputs, **kwargs):
        left_input = inputs[0]
        mid_input = inputs[1]
        right_input = inputs[2]

        stacked_left_input = stack_input(left_input, int(left_input.shape[1] / 8), 8)
        stacked_mid_input = stack_input(mid_input, int(mid_input.shape[1] / 8), 8)
        stacked_right_input = stack_input(right_input, int(right_input.shape[1] / 8), 8)

        conv1_1 = self.conv1_1(stacked_left_input)
        conv1_2 = self.conv1_2(stacked_mid_input)
        conv1_3 = self.conv1_3(stacked_right_input)

        pool_1 = self.pool_1(conv1_1)
        pool_2 = self.pool_2(conv1_2)
        pool_3 = self.pool_3(conv1_3)

        # # https://stackoverflow.com/questions/49313650/how-could-i-get-both-the-final-hidden-state-and-sequence-in-a-lstm-layer-when-us
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

        # attention layer takes query and values.
        # In an LSTM, the query is the hidden (cell state) and values is output (hidden state)
        left_context = self.left_attention([state_1_c, state_1_h])
        mid_sentence = self.mid_attention([state_2_c, state_2_h])
        right_context = self.right_attention([state_3_c, state_3_h])

        concatenated_layer = Concatenate(name="fully_connected")(
            [left_context, mid_sentence, right_context]
        )

        dense = self.dense_output(concatenated_layer)

        return dense
