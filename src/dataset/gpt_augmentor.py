import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, GPT2TokenizerFast

from typing import List


class Augmentor:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_fast = GPT2TokenizerFast.from_pretrained("gpt2")

    gpt_model = TFGPT2LMHeadModel.from_pretrained(
        "gpt2", pad_token_id=tokenizer.eos_token_id
    )

    # https://gist.github.com/GeorgeDittmar/5c57a35332b2b5818e51618af7953351
    @classmethod
    def augment_gpt2(
        cls,
        sentences: List,
        fast=False,
        num_return_sequences=3,
        max_seq_word_length=200,
        verbose=False,
    ):
        """creates segments augmented based on gpt2

        Args:
            sentences (List): all the text sentences in list format
            fast (bool, optional): Use the fast tokenizer. Defaults to False.
            num_return_sequences (int, optional): How many different augmentations to return. Defaults to 3.
            max_seq_word_length (int, optional): How large each segment will be in terms of words. Defaults to 50.

        Returns:
            List[List]: a list of lists with augmented segments.
        """
        generated_segments = []
        tokenizer = cls.tokenizer if not fast else cls.tokenizer_fast

        for i, sentence in enumerate(sentences):
            # encode context the generation is conditioned on
            input_ids = tokenizer.encode(sentence, return_tensors="tf")

            # set seed to reproduce results. Feel free to change the seed though to get different results
            tf.random.set_seed(32)

            # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
            sample_outputs = cls.gpt_model.generate(
                input_ids,
                do_sample=True,
                max_length=max_seq_word_length,
                top_k=10,
                temperature=0.7,
                no_repeat_ngram_size=2,
                num_return_sequences=num_return_sequences,
            )

            generated_segments.append(
                [tokenizer.decode(x, skip_special_tokens=True)
                 for x in sample_outputs]
            )

            if verbose:
                print(f"Completed augmenting {i+1}/{len(sentences)}...")

        return generated_segments
