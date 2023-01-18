import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, GPT2TokenizerFast

from src.dataset.utils import (
    truncate_by_token,
    avg_segment_length_by_char,
    avg_segment_length_by_token,
)

from nltk.tokenize import word_tokenize

from typing import List


class Augmentor:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_fast = GPT2TokenizerFast.from_pretrained("gpt2")

    gpt_model = TFGPT2LMHeadModel.from_pretrained(
        "gpt2", pad_token_id=tokenizer.eos_token_id
    )

    @classmethod
    def post_augmentation_processing(cls, sentence, real_sentence_chars):
        """strips the original sentence from the outputted augmented sentence

        Args:
            sentence (str): the augmented sentence
            real_sentence_chars (int): the number of chars the original sentence consists of

        Returns:
            str: the augmented sentence
        """
        return sentence[real_sentence_chars:]

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

    @classmethod
    def augment_gpt2_single(
        cls,
        sentence: str,
        fast=False,
        num_return_sequences=3,
        output_tokens=200,
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
        # encode context the generation is conditioned on
        input_ids = tokenizer.encode(sentence, return_tensors="tf")

        # set seed to reproduce results. Feel free to change the seed though to get different results
        tf.random.set_seed(32)

        # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
        sample_outputs = cls.gpt_model.generate(
            input_ids,
            do_sample=True,
            # corresponds to all the new tokens appended to the input
            max_new_tokens=output_tokens,
            top_k=10,
            temperature=0.7,
            no_repeat_ngram_size=2,
            num_return_sequences=num_return_sequences,
        )

        generated_segments.append(
            [tokenizer.decode(x, skip_special_tokens=True)
             for x in sample_outputs]
        )

        print("completed augmentation...")

        return generated_segments

    """
        - Using GPT-2, we take the first truncated portion of the first sentence in a segment and feed it into the model. The output should be the same size as the the overall dataset average sentence length.
        - We then take that output sentence as the first sentence in the augmented segment
        - Using that newly augmented sentence, we feed it into GPT again to generate a new sentence of the same size.
        - We do this autoregressive process for n times.
            - For experimentation, we do n = k/2 where k is the average segment size in the dataset.
        - On average, we will have about half the amount of total data in our augmented dataset than our real dataset

        Note: By default, GPT is autoregressive, so instead of having to re-run GPT on every sentence that's generated, just run it once and multiply the output_tokens by n to get the desired sentences. Afterward, the post-processing will need to chop the initial sentence off and break the complete output sentence into its relative sentences.
    """
    @classmethod
    def gta1(cls, segments: List[List[str]], min_sent_tokens: int = 64, max_sent_tokens: int = 64) -> List[List[str]]:
        """
        Args:
            segments (List[List[str]]): List of segments
            max_sent_tokens (int, optional): Max number of words your real sentences will have before feeding into GPT2. Defaults to 64.
            min_sent_tokens (int, optional): Min number of words your real sentences will have before feeding into GPT2. Defaults to 64.
                - Note: the min sent tokens is to avoid sentences like "Um.", "Okay, that's good" for example where there's no context.

        Returns:
            List[List[str]]: returns a list of segments with the same shape as `segments`
        """
        dataset_avg_sentence_length = sum(
            [avg_segment_length_by_token(segment, floor=True)
             for segment in segments]
        ) // len(segments)

        avg_segment_length = sum([len(t) for t in segments]) // len(
            segments
        )  # avg number of sentences per segment in the dataset
        # the number of sentences we will generate per segment
        n = int(avg_segment_length / 2)

        augmented_segments = []

        print("beginning GTA 1 augmentation.")
        for i, segment in enumerate(segments):
            augmented_segment = []
            next_sentence = segment[0]
            for sentence in segment:
                if len(word_tokenize(sentence)) >= min_sent_tokens:
                    next_sentence = sentence
                    break
            for j in range(0, avg_segment_length):
                next_sentence = (
                    next_sentence if len(
                        augmented_segment) == 0 else augmented_segment[-1]
                )
                next_sentence = truncate_by_token(
                    next_sentence, max_sent_tokens)
                next_sentence_length = len(next_sentence)
                sentence_tokens_length = len(word_tokenize(next_sentence))
                # create segment
                augmented_sentence = Augmentor.augment_gpt2_single(
                    next_sentence,
                    fast=True,
                    # add the length of the current sentence to the dataset avg length of sentence
                    output_tokens=int(n * int(dataset_avg_sentence_length)),
                    num_return_sequences=1,
                )

                augmented_sentence = cls.post_augmentation_processing(
                    # feed in the first generated sentence
                    augmented_sentence[0][0],
                    next_sentence_length,
                )

                augmented_segment.append(augmented_sentence)
                print(".", end="")

            print(f"completed {i+1}/{len(segments)} segments")
            augmented_segments.append(augmented_segment)

        return augmented_segments

    """
        - Using GPT-2, we take the first truncated portion of the first sentence in a segment and feed it into the model. The output should be the same size as the first sentence length (for averaging similar segment sizes).
        - That first outputted sentence becomes the target sentence for the augmented segment.
        - Using the sentence sentence in the real segment, we repeat the first step. The second sentence in the augmented segment will be the output of the real second sentence fed into GPT-2.
        - Continuing this process, we will be left with an augmented segment the same exact size as the real segment it’s modeled after with hopefully less variance than GTA 1 toward the end of the segments.
        
        Cons:

        - Possible disjointedness with augmented sentences since they may vary quite a bit from the immediate sentence previously due to relying on an intermediary in-between sentence to generate.
    """
    @classmethod
    def gta2(cls, segments: List[List[str]], min_sent_tokens: int = 64, max_sent_tokens: int = 64) -> List[List[str]]:
        """
        Args:
            segments (List[List[str]]): real segments to feed into GPT2
            max_sent_tokens (int, optional): Max number of words your real sentences will have before feeding into GPT2. Defaults to 64.
            min_sent_tokens (int, optional): Min number of words your real sentences will have before feeding into GPT2. Defaults to 64.
                - Note: the min sent tokens is to avoid sentences like "Um.", "Okay, that's good" for example where there's no context.

        Returns:
            List[List[str]]: returns a list of segments with the same shape as `segments`
        """
        dataset_avg_sentence_length = sum(
            [avg_segment_length_by_token(segment, floor=True)
             for segment in segments]
        ) // len(segments)

        max_sent_tokens = 64
        augmented_segments = []

        print("beginning GTA 2 augmentation.")
        for i, segment in enumerate(segments):
            augmented_segment = []
            next_sentence = segment[0]
            for sentence in segment:
                if len(word_tokenize(sentence)) >= min_sent_tokens:
                    next_sentence = sentence
                    break
            for sentence in segment:
                sentence = truncate_by_token(sentence, max_sent_tokens)
                sentence_length = len(sentence)

                augmented_sentence = Augmentor.augment_gpt2_single(
                    sentence,
                    fast=True,
                    # we want to generate an avg number of tokens per sentence
                    output_tokens=int(dataset_avg_sentence_length),
                    num_return_sequences=1,
                )

                augmented_sentence = cls.post_augmentation_processing(
                    # feed in the first generated sentence
                    augmented_sentence[0][0],
                    sentence_length,
                )

                augmented_segment.append(augmented_sentence)
                print(".", end="")

            print(f"completed {i+1}/{len(segments)} segments")
            augmented_segments.append(augmented_segment)

        return augmented_segments
