import tensorflow as tf

from tensorflow.keras.layers import Concatenate

from .context_encoder import ContextEncoderMixin, pool_output


class ContextEncoder(tf.keras.Model, ContextEncoderMixin):
    def __init__(self, *inputs, **kwargs):
        super().__init__()
        self.output_bias = None

        self.init_params(
            [
                "lstm_dropout",
                "lstm_size",
                "final_dropout",
                "dense_neurons",
                "output_bias",
                "bert_trainable",
                "bert_type",
            ],
            kwargs,
        )

        self.init_bert_stack(self.bert_type)

        self.init_dense_stack()

        self.init_output_stack()

        self.init_dropout_stack()

    def get_inputs(self, inputs):
        if self.bert_type.startswith("distilbert"):
            bert_left_output = pool_output(self.bert(inputs[0])[0])
            bert_mid_output = pool_output(self.bert(inputs[1])[0])
            bert_right_output = pool_output(self.bert(inputs[2])[0])
        elif self.bert_type.startswith("albert"):
            # return the -1th as albert provides that as the pooled output
            bert_left_output = self.bert(inputs[0])[-1]
            bert_mid_output = self.bert(inputs[1])[-1]
            bert_right_output = self.bert(inputs[2])[-1]
        else:
            # return the -1th as albert provides that as the pooled output
            bert_left_output = self.bert(inputs[0])[0]
            bert_mid_output = self.bert(inputs[1])[0]
            bert_right_output = self.bert(inputs[2])[0]
        return bert_left_output, bert_mid_output, bert_right_output

    def call(self, inputs, **kwargs):

        left_input, mid_input, right_input = self.get_inputs(inputs)

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
