from nltk import word_tokenize

# tokneize a sent


def tokenize(sent):
    tokens = " ".join(word_tokenize(sent.lower()))
    return tokens

# filter some noises caused by speech recognition


def clean_sentence(text):
    text = text.replace("{vocalsound}", "")
    text = text.replace("{disfmarker}", "")
    text = text.replace("a_m_i_", "ami")
    text = text.replace("l_c_d_", "lcd")
    text = text.replace("p_m_s", "pms")
    text = text.replace("t_v_", "tv")
    text = text.replace("{pause}", "")
    text = text.replace("{nonvocalsound}", "")
    text = text.replace("{gap}", "")
    return text


def preprocess_text_segmentation(data):
    # shape: [["sentence 1", "sentence 2"], ["sentence3", "sentence4"]]
    text_segments = []
    for sample in data:
        # get the topic list and meeting transcript from the current sample
        topic_list = sample["topic_list"]
        meeting_transcripts = sample["meeting_transcripts"]

        for topic in topic_list:
            prev_end_span = -1
            for text_span in topic["relevant_text_span"]:
                # for this specific topic, get the relevant start and end spans
                start_of_span = int(text_span[0])
                end_of_span = int(text_span[-1])

                # MISSING TOPIC CASE
                # this means that there are some sentences without a topic
                # for example, [0, 19] and [24, 29], where 20 - 23 are missing
                if start_of_span != (prev_end_span + 1):
                    missing_topic_meeting = meeting_transcripts[
                        prev_end_span + 1: start_of_span - 1
                    ]
                    if len(missing_topic_meeting) != 0:
                        missing_topic_full_segment = []
                        for segment in missing_topic_meeting:
                            content = segment["content"]
                            # clean the text before insertion and tokenize
                            cleaned_content = clean_sentence(content)
                            tokenized_content = tokenize(cleaned_content)
                            # get the content for the segment and throw it into the list of segments
                            missing_topic_full_segment.append(
                                tokenized_content)
                        text_segments.append(missing_topic_full_segment)

                # REAL TOPIC CASE
                topic_meeting = meeting_transcripts[start_of_span:end_of_span]
                if "meeting" not in topic:
                    topic["meeting"] = []
                topic["meeting"].append(topic_meeting)

                full_segment = []
                for segment in topic_meeting:
                    content = segment["content"]
                    # clean the text before insertion and tokenize
                    cleaned_content = clean_sentence(content)
                    tokenized_content = tokenize(cleaned_content)
                    # get the content for the segment and throw it into the list of segments
                    full_segment.append(tokenized_content)

                text_segments.append(full_segment)

    return text_segments


def format_data_for_db_insertion(data):
    tuples = []

    id = 1
    # id of the current parent as iterating through
    parent_id = None
    for i, segment in enumerate(data):
        for segment_index, sentence in enumerate(segment):
            # start of the segment
            if segment_index == 0:
                # (autoint id, sentence string, target [1 or 0], parent, sequence)
                current_sentence = (sentence, 1, None, segment_index)
                tuples.append(current_sentence)
                parent_id = id
            # every other sentence in the segment
            else:
                # (autoint id, sentence string, target [1 or 0], parent, sequence)
                current_sentence = (sentence, 0, parent_id, segment_index)
                tuples.append(current_sentence)
            id += 1

    return tuples
