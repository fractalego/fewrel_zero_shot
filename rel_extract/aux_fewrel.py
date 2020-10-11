import copy
import json
import random

import torch
from tqdm import tqdm


def start_from_target(target, label):
    for index, item in enumerate(target):
        if item == label:
            return index
    if item == label:
        return index
    return -1


def end_from_target(target, label):
    old_label = ''
    for index, item in enumerate(target):
        if item != label and old_label == label:
            return index
        old_label = item
    if item == label:
        return index + 1
    return -1


def get_all_sentences_and_relations_from_json(filename, max_lines=-1, masked_ratio=0):
    json_data = json.load(open(filename))

    sentences = []
    relations = []
    for relation in json_data.keys():
        for rel_dict in list(json_data[relation])[:max_lines]:
            tokens = rel_dict['tokens']

            subject_indices = rel_dict['h'][2][0]
            object_indices = rel_dict['t'][2][0]

            items = []

            for index, token in enumerate(tokens):
                if random.uniform(0, 1) < masked_ratio:
                    token = '[MASK]'

                if index in subject_indices:
                    items.append((token, 'SUBJECT'))

                if index in object_indices:
                    items.append((token, 'OBJECT'))

                items.append((token, ''))

            sentences.append(items)
            relations.append(relation)

    return sentences, relations


def get_sentences_and_targets_from_sentence_tuples(tuples_list):
    all_sentences = []
    all_targets = []
    for tuple in tuples_list:
        sentence = ''
        for item in tuple:
            sentence += item[0] + ' '
        all_sentences.append(sentence[:-1])
        all_targets.append([item[1] for item in tuple])
    return all_sentences, all_targets


def create_one_hot_vector(index, lenght):
    vector = [0.] * lenght
    vector[index] = 1.
    return vector


def get_new_targets(sentence, targets, tokenizer):
    new_targets = []
    for word, target in zip(sentence.split(), targets):
        new_tokens = tokenizer.tokenize(word)
        if len(new_tokens) == 1:
            new_targets.append(target)
            continue
        new_targets.append(target)
        for _ in new_tokens[1:]:
            new_targets.append(target)
    return new_targets


def get_data_from_tuples_with_description_and_label(tuples_list, relations, tokenizer, rel_map):
    all_data = []
    sentences, targets = get_sentences_and_targets_from_sentence_tuples(tuples_list)
    for sentence, target, relation in tqdm(zip(sentences, targets, relations), total=len(sentences)):
        candidates = [rel_map[relation]['sentence']] + [rel_map[relation]['label']] + rel_map[relation]['aliases']
        relation_sentence = random.choice(candidates)

        input_ids = torch.tensor([[101] + tokenizer.encode(relation_sentence, add_special_tokens=False)
                                  + [102] + tokenizer.encode(sentence, add_special_tokens=False)
                                  + [102]
                                  ])

        relation_sentence_length = len(tokenizer.encode(relation_sentence, add_special_tokens=False)) + 1

        target = get_new_targets(sentence, target, tokenizer)
        length = len(target) + 2
        subj_start = start_from_target(target, 'SUBJECT') + 1
        obj_start = start_from_target(target, 'OBJECT') + 1
        subj_end = end_from_target(target, 'SUBJECT') + 1
        obj_end = end_from_target(target, 'OBJECT') + 1

        subj_start_label = torch.tensor(create_one_hot_vector(subj_start, length))
        obj_start_label = torch.tensor(create_one_hot_vector(obj_start, length))
        subj_end_label = torch.tensor(create_one_hot_vector(subj_end, length))
        obj_end_label = torch.tensor(create_one_hot_vector(obj_end, length))

        all_data.append((input_ids, subj_start_label, subj_end_label, obj_start_label, obj_end_label,
                         relation_sentence_length))

    return all_data


def get_negative_data_from_tuples_with_label_and_aliases(tuples_list, relations, tokenizer, rel_map, all_relations,
                                                         max_labels):
    all_data = []
    negative_relations = []
    sentences, targets = get_sentences_and_targets_from_sentence_tuples(tuples_list)
    for sentence, target, relation in tqdm(zip(sentences, targets, relations), total=len(sentences)):
        candidates = copy.deepcopy(all_relations)
        try:
            candidates.remove(relation)
        except:
            pass
        random.shuffle(candidates)
        properties = candidates[:max_labels]
        for property in properties:
            negative_relations.append(property)
            value = rel_map[property]

            relation_sentence = random.choice([value['sentence']] + [value['label']] + value['aliases'])
            if relation_sentence == {}:
                relation_sentence = rel_map[property]['label']

            input_ids = torch.tensor([[101] + tokenizer.encode(relation_sentence, add_special_tokens=False)
                                      + [102] + tokenizer.encode(sentence, add_special_tokens=False)
                                      + [102]
                                      ])

            relation_sentence_length = len(tokenizer.encode(relation_sentence, add_special_tokens=False)) + 1

            target = get_new_targets(sentence, target, tokenizer)
            length = len(target) + 2
            subj_start, obj_start, subj_end, obj_end = 0, 0, 0, 0

            subj_start_label = torch.tensor(create_one_hot_vector(subj_start, length))
            obj_start_label = torch.tensor(create_one_hot_vector(obj_start, length))
            subj_end_label = torch.tensor(create_one_hot_vector(subj_end, length))
            obj_end_label = torch.tensor(create_one_hot_vector(obj_end, length))

            all_data.append((input_ids, subj_start_label, subj_end_label, obj_start_label, obj_end_label,
                             relation_sentence_length))

    return all_data, negative_relations


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def batchify(data, n):
    len_dict = {}
    for item in data:
        in_length = item[0].shape[1]
        out_length = item[5]
        try:
            len_dict[(in_length, out_length)].append(item)
        except:
            len_dict[(in_length, out_length)] = [item]

    batch_chunks = []
    for k in len_dict.keys():
        vectors = len_dict[k]
        batch_chunks += chunks(vectors, n)

    batches = []
    for chunk in batch_chunks:
        input = torch.stack([item[0][0] for item in chunk])
        labels1 = torch.stack([item[1] for item in chunk])
        labels2 = torch.stack([item[2] for item in chunk])
        labels3 = torch.stack([item[3] for item in chunk])
        labels4 = torch.stack([item[4] for item in chunk])
        labels5 = torch.tensor([item[5] for item in chunk])
        batches.append((input, labels1, labels2, labels3, labels4, labels5))

    return batches


def erase_non_entities(all_words, all_entities, all_idx):
    return [(w, e, i) for w, e, i in zip(all_words, all_entities, all_idx) if e and w != ' ']


def join_consecutive_tuples(tuples):
    for i in range(len(tuples) - 1):
        curr_type = tuples[i][1]
        curr_end_idx = tuples[i][2][1]
        next_type = tuples[i + 1][1]
        next_start_idx = tuples[i + 1][2][0]
        if curr_type == next_type and curr_end_idx == next_start_idx - 1:
            curr_word = tuples[i][0]
            next_word = tuples[i + 1][0]
            curr_start_idx = tuples[i][2][0]
            next_end_idx = tuples[i + 1][2][1]
            tuples[i + 1] = (curr_word + ' ' + next_word,
                             curr_type,
                             (curr_start_idx, next_end_idx))
            tuples[i] = ()
    tuples = [t for t in tuples if t]
    return tuples


def load_rel_dict_for_fewrel(filename):
    rel_dict = {}
    json_data = json.load(open(filename))
    for item in json_data:
        rel_dict[item['id']] = {'sentence': item['description'],
                                'label': item['label'],
                                'aliases': item['aliases'],
                                }

    return rel_dict


def get_discriminative_data_from_tuples(tuples_list, relations, tokenizer, rel_map, masked_ratio=0):
    all_data = []
    sentences, targets = get_sentences_and_targets_from_sentence_tuples(tuples_list)
    for sentence, target, relation in tqdm(zip(sentences, targets, relations), total=len(sentences)):
        target = get_new_targets(sentence, target, tokenizer)
        subj_start = start_from_target(target, 'SUBJECT') + 1
        obj_start = start_from_target(target, 'OBJECT') + 1
        subj_end = end_from_target(target, 'SUBJECT') + 1
        obj_end = end_from_target(target, 'OBJECT') + 1

        all_data.append((sentence, relation, subj_start, subj_end, obj_start, obj_end))

    return all_data
