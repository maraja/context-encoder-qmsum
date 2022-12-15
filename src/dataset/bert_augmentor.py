import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

from typing import List


class Augmentor:
    bert_sub_aug = naw.ContextualWordEmbsAug(
        model_path="bert-base-uncased", action="substitute"
    )
    bert_insert_aug = naw.ContextualWordEmbsAug(
        model_path="bert-base-uncased", action="insert"
    )

    @classmethod
    def augment_bert(cls, sentences: List, augment_type: str):
        if augment_type == "substitute":
            augmented_sentences = cls.bert_sub_aug.augment(sentences)
            return sentences, augmented_sentences
        elif augment_type == "insert":
            augmented_sentences = cls.bert_insert_aug.augment(sentences)
            return sentences, augmented_sentences
        else:
            raise ValueError("augment_type must be either substitute or insert.")
