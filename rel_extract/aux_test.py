import copy
import os
import random
import sys

import torch
from tqdm import tqdm

from rel_extract.aux_fewrel import load_rel_dict_for_fewrel, get_discriminative_data_from_tuples, \
    get_all_sentences_and_relations_from_json

_path = os.path.dirname(__file__)
_rel_dict_filename = os.path.join(_path, '../data/props.json')
_dev_filename = os.path.join(_path, '../data/val_wiki.json')
rel_dict = load_rel_dict_for_fewrel(_rel_dict_filename)
sentences, relations = get_all_sentences_and_relations_from_json(_dev_filename)
all_relations = list(set(relations))


def test_with_full_match(model, tokenizer, n_ways):
    dev_data = get_discriminative_data_from_tuples(sentences, relations, tokenizer, rel_dict)
    random.shuffle(dev_data)

    count_dict = {}

    for tuple in tqdm(dev_data):
        sentence, original_relation, subj_start_target, subj_end_target, obj_start_target, obj_end_target = tuple
        discriminative_model_relation = ''
        old_adversarial_score = 1

        candidate_relations = copy.deepcopy(all_relations)
        candidate_relations.remove(original_relation)
        random.shuffle(candidate_relations)
        candidate_relations = [original_relation] + candidate_relations[:n_ways - 1]
        random.shuffle(candidate_relations)
        for new_relation in candidate_relations:

            is_positive_match, adversarial_score = \
                run_generative_full_matches(model,
                                            rel_dict,
                                            tokenizer,
                                            sentence,
                                            original_relation,
                                            new_relation,
                                            subj_start_target,
                                            subj_end_target,
                                            obj_start_target,
                                            obj_end_target)

            if adversarial_score < old_adversarial_score:
                discriminative_model_relation = new_relation
                old_adversarial_score = adversarial_score

        result = ''

        if discriminative_model_relation == '':
            result = 'fn'
        elif discriminative_model_relation == original_relation:
            result = 'tp'
        elif discriminative_model_relation != original_relation:
            result = 'fp'

        if result in count_dict:
            count_dict[result] += 1
        else:
            count_dict[result] = 1

    tp = count_dict['tp'] if 'tp' in count_dict else 0
    fp = count_dict['fp'] if 'fp' in count_dict else 0

    precision = 0
    if tp or fp:
        precision = tp / (tp + fp)

    print(count_dict)
    print('   accuracy:', precision)
    sys.stdout.flush()


def run_model(model, tokenizer, rel_map, sentence, relation):
    relation_sentence = rel_map[relation]['sentence']

    inputs = torch.tensor([[101] + tokenizer.encode(relation_sentence, add_special_tokens=False)
                           + [102] + tokenizer.encode(sentence, add_special_tokens=False)
                           + [102]
                           ])

    length = torch.tensor([len(tokenizer.encode(relation_sentence, add_special_tokens=False)) + 1])
    subj_starts, subj_ends, obj_starts, obj_ends = model(inputs.cuda(), length)

    return subj_starts[0], subj_ends[0], obj_starts[0], obj_ends[0], inputs[0]


def run_generative_full_matches(model,
                                rel_map,
                                tokenizer,
                                sentence,
                                original_relation,
                                relation,
                                subj_start_target,
                                subj_end_target,
                                obj_start_target,
                                obj_end_target):
    subj_start_ohv, subj_end_ohv, obj_start_ohv, obj_end_ohv, inputs = run_model(model,
                                                                                 tokenizer,
                                                                                 rel_map, sentence,
                                                                                 relation)
    words = [tokenizer.convert_ids_to_tokens([i])[0] for i in list(inputs)[-len(subj_end_ohv):]]

    subj_start = torch.argmax(subj_start_ohv)
    subj_end = torch.argmax(subj_end_ohv)
    obj_start = torch.argmax(obj_start_ohv)
    obj_end = torch.argmax(obj_end_ohv)

    model_has_candidates = False
    threshold = 1

    adversarial_score = min(subj_start_ohv[0], subj_end_ohv[0], obj_start_ohv[0], obj_end_ohv[0])
    if adversarial_score < threshold:
        subj_start = torch.argmax(subj_start_ohv[1:]) + 1
        subj_end = torch.argmax(subj_end_ohv[1:]) + 1
        obj_start = torch.argmax(obj_start_ohv[1:]) + 1
        obj_end = torch.argmax(obj_end_ohv[1:]) + 1
        model_has_candidates = True

    model_says_it_is_positive = False
    if model_has_candidates \
            and words[subj_start:subj_end] == words[subj_start_target:subj_end_target] \
            and words[obj_start:obj_end] == words[obj_start_target:obj_end_target]:
        model_says_it_is_positive = True

    return model_says_it_is_positive, adversarial_score
