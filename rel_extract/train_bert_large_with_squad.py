import os
import random
import sys

import torch
from torch import nn
from transformers import BertModel, BertTokenizer

from rel_extract.aux_fewrel import batchify, get_all_sentences_and_relations_from_json, \
    load_rel_dict_for_fewrel, get_data_from_tuples_with_description_and_label, \
    get_negative_data_from_tuples_with_label_and_aliases
from rel_extract.aux_test import test_with_full_match
from rel_extract.aux_train import train, select_sample
from rel_extract.model import RelTaggerModel

_path = os.path.dirname(__file__)
_train_filename = os.path.join(_path, '../data/train_wiki.json')
_rel_dict_filename = os.path.join(_path, '../data/props.json')
_all_relations_file = os.path.join(_path, '../data/all_train_relations.txt')
_save_filename = os.path.join(_path, '../data/save_bert_large_with_squad')

MODEL = (BertModel, BertTokenizer, 'bert-large-uncased-whole-word-masking-finetuned-squad')

if __name__ == '__main__':
    model_class, tokenizer_class, pretrained_weights = MODEL
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    language_model = model_class.from_pretrained(pretrained_weights)

    rel_dict = load_rel_dict_for_fewrel(_rel_dict_filename)
    all_relations = eval(open(_all_relations_file).read())

    print('Loading training data')
    sentences, relations = get_all_sentences_and_relations_from_json(_train_filename)
    train_data = get_data_from_tuples_with_description_and_label(sentences, relations, tokenizer, rel_dict)
    sample_sentences, sample_relations = select_sample(sentences, relations, 1.)
    train_data += \
        get_negative_data_from_tuples_with_label_and_aliases(sample_sentences, sample_relations, tokenizer, rel_dict,
                                                             all_relations, max_labels=2)[0]
    random.shuffle(train_data)
    train_batches = batchify(train_data, 16)

    train_model = RelTaggerModel(language_model)
    train_model.cuda()

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(train_model.parameters(), lr=3e-6)

    for epoch in range(10):
        random.shuffle(train_batches)
        train_model.train()
        loss = train(train_model, train_batches, optimizer, criterion)
        print('Epoch:', epoch, 'Loss:', loss)
        test_with_full_match(train_model, tokenizer, n_ways=5)

        torch.save({
            'epoch': epoch,
            'model_state_dict': train_model.state_dict()},
            _save_filename + str(epoch))

        sys.stdout.flush()
