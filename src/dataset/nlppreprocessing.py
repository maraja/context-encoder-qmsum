import unicodedata
import re
import tensorflow as tf

from typing import List
import pkg_resources
from stop_words import get_stop_words
from symspellpy import SymSpell, Verbosity
from language_detector import detect_language
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


###################################
#### sentence level preprocess ####
###################################


def f_base_simple(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """
    # normalization 2: lower case
    s = s.lower()
    # normalization 6: string * as delimiter
    s = re.sub(r"\*|\W\*|\*\W", ". ", s)
    # normalization 7: stuff in parenthesis, assumed to be less informal
    s = re.sub(r"\(.*?\)", ". ", s)

    return s.strip()


# lowercase + base filter
# some basic normalization
def f_base(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """
    # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
    s = re.sub(r"([a-z])([A-Z])", r"\1\. \2", s)  # before lower case
    # normalization 2: lower case
    s = s.lower()
    # normalization 3: "&gt", "&lt"
    s = re.sub(r"&gt|&lt", " ", s)
    # normalization 4: letter repetition (if more than 2)
    s = re.sub(r"([a-z])\1{2,}", r"\1", s)
    # normalization 5: non-word repetition (if more than 1)
    s = re.sub(r"([\W+])\1{1,}", r"\1", s)
    # normalization 6: string * as delimiter
    s = re.sub(r"\*|\W\*|\*\W", ". ", s)
    # normalization 7: stuff in parenthesis, assumed to be less informal
    s = re.sub(r"\(.*?\)", ". ", s)
    # normalization 8: xxx[?!]. -- > xxx.
    s = re.sub(r"\W+?\.", ".", s)
    # normalization 9: [.?!] --> [.?!] xxx
    s = re.sub(r"(\.|\?|!)(\w)", r"\1 \2", s)
    # normalization 10: ' ing ', noise text
    s = re.sub(r" ing ", " ", s)
    # normalization 11: noise text
    s = re.sub(r"product received for free[.| ]", " ", s)
    # normalization 12: phrase repetition
    s = re.sub(r"(.{2,}?)\1{1,}", r"\1", s)

    return s.strip()


# language detection
def f_lan(s):
    """
    :param s: string to be processed
    :return: boolean (s is English)
    """

    # some reviews are actually english but biased toward french
    return detect_language(s) in {"English", "French"}


###############################
#### word level preprocess ####
###############################

# filtering out punctuations and numbers
def f_punct(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with punct and number filter out
    """
    return [word for word in w_list if word.isalpha()]


# selecting nouns
def f_noun(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with only nouns selected
    """
    return [word for (word, pos) in nltk.pos_tag(w_list) if pos[:2] == "NN"]


# typo correction
def f_typo(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with typo fixed by symspell. words with no match up will be dropped
    """
    w_list_fixed = []
    for word in w_list:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
        if suggestions:
            w_list_fixed.append(suggestions[0].term)
        else:
            pass
            # do word segmentation, deprecated for inefficiency
            # w_seg = sym_spell.word_segmentation(phrase=word)
            # w_list_fixed.extend(w_seg.corrected_string.split())
    return w_list_fixed


# stemming if doing word-wise
p_stemmer = PorterStemmer()


def f_stem(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with stemming
    """
    return [p_stemmer.stem(word) for word in w_list]


# filtering out stop words
# create English stop words list
en_stop = get_stop_words("en")
# en_stop.append('game')
# en_stop.append('play')
# en_stop.append('player')
# en_stop.append('time')


def f_stopw(w_list):
    """
    filtering out stop words
    """
    return [word for word in w_list if word not in en_stop]


def preprocess_sent_simple(rw):
    """
    Get sentence level preprocessed data from raw review texts
    :param rw: review to be processed
    :return: sentence level pre-processed review
    """
    return f_base(rw)


def preprocess_sent(rw):
    """
    Get sentence level preprocessed data from raw review texts
    :param rw: review to be processed
    :return: sentence level pre-processed review
    """
    s = f_base(rw)
    if not f_lan(s):
        return None
    return s


def preprocess_word(s):
    """
    Get word level preprocessed data from preprocessed sentences
    including: remove punctuation, select noun, fix typo, stem, stop_words
    :param s: sentence to be processed
    :return: word level pre-processed review
    """
    if not s:
        return None
    w_list = word_tokenize(s)
    w_list = f_punct(w_list)
    w_list = f_noun(w_list)
    # w_list = f_typo(w_list)
    w_list = f_stem(w_list)
    w_list = f_stopw(w_list)

    return w_list


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    # w = '<start> ' + w + ' <end>'
    return w


def augment_sentences_and_labels(
    sentence_segments,
    label_segments,
    augment_with_target_sent_duplication=False,
    max_segment_length=5,
):
    temp_segment = []
    augmented_sentence_segments = []
    augmented_label_segments = []
    for i, segment in enumerate(sentence_segments):
        if label_segments[i][0] != 1:
            augmented_sentence_segments.append(segment[:max_segment_length])
            continue
        # reset the counter
        for sent in segment:
            temp_segment.append(sent)
            if len(temp_segment) == max_segment_length:
                augmented_sentence_segments.append(temp_segment)
                temp_segment = []
                # append the classification to the first slot
                if augment_with_target_sent_duplication:
                    temp_segment.append(segment[0])
        augmented_sentence_segments.append(temp_segment)
        temp_segment = []

    temp_segment = []
    for i, segment in enumerate(label_segments):
        if label_segments[i][0] != 1:
            augmented_label_segments.append(segment[:max_segment_length])
            continue
        # reset the counter
        for l in segment:
            if len(temp_segment) == 0:
                temp_segment = [1]
            else:
                temp_segment.append(l)
            if len(temp_segment) == max_segment_length:
                augmented_label_segments.append(temp_segment)
                temp_segment = []
                # append the classification to the first slot
                if augment_with_target_sent_duplication:
                    temp_segment.append(1)
        augmented_label_segments.append(temp_segment)
        temp_segment = []

    return augmented_sentence_segments, augmented_label_segments


def shuffle_sentences_and_labels(sentence_segments, label_segments):
    # shuffle our data
    import random

    c = list(zip(sentence_segments, label_segments))

    random.shuffle(c)

    sentence_segments, label_segments = zip(*c)
    return sentence_segments, label_segments


def split_to_segments(sentences: List, labels: List) -> List[List[str]]:
    sentence_segments = []
    label_segments = []
    current_sentence_segment = []
    current_label_segment = []
    for sentence, label in zip(sentences, labels):
        if label == 0:
            # continue appending to buffer
            current_sentence_segment.append(sentence)
            current_label_segment.append(label)
        elif label == 1:
            if len(current_sentence_segment) > 0 and len(current_label_segment) > 0:
                # append the segment to full list
                sentence_segments.append(current_sentence_segment)
                label_segments.append(current_label_segment)
            # empty out the current segment buffers
            current_sentence_segment = []
            current_label_segment = []
            # add the beginning of the new segment to the buffers
            current_sentence_segment.append(sentence)
            current_label_segment.append(label)

    return sentence_segments, label_segments


def tokenize(lang, max_input_length):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tensor, padding="post", maxlen=max_input_length
    )

    return tensor, lang_tokenizer


def preprocess_dataset(df, max_input_length=None):
    left_tensor, left_tensor_tokenizer = tokenize(
        df["left"].tolist(), max_input_length=max_input_length
    )
    mid_tensor, mid_tensor_tokenizer = tokenize(
        df["middle"].tolist(), max_input_length=max_input_length
    )
    right_tensor, right_tensor_tokenizer = tokenize(
        df["right"].tolist(), max_input_length=max_input_length
    )

    return (
        left_tensor,
        left_tensor_tokenizer,
        mid_tensor,
        mid_tensor_tokenizer,
        right_tensor,
        right_tensor_tokenizer,
    )

