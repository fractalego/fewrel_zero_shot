import os

import torch
from transformers import BertModel, BertTokenizer

from rel_extract.aux_test import test_with_full_match
from rel_extract.model import RelTaggerModel

_path = os.path.dirname(__file__)
_pre_trained_filename = os.path.join(_path, '../data/save_bert_large_with_squad0')
MODEL = (BertModel, BertTokenizer, 'bert-large-uncased-whole-word-masking-finetuned-squad')

model_class, tokenizer_class, pretrained_weights = MODEL
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
language_model = model_class.from_pretrained(pretrained_weights)

if __name__ == '__main__':
    model = RelTaggerModel(language_model)
    checkpoint = torch.load(_pre_trained_filename, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    test_with_full_match(model, tokenizer, n_ways=10)
