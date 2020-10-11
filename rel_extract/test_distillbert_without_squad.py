import os

import torch
from transformers import DistilBertModel, DistilBertTokenizer

from rel_extract.aux_test import test_with_full_match
from rel_extract.model_distillbert import RelTaggerModel

_path = os.path.dirname(__file__)
_pre_trained_filename = os.path.join(_path, '../data/save_distillbert_without_squad1')
MODEL = (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased')

model_class, tokenizer_class, pretrained_weights = MODEL
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
language_model = model_class.from_pretrained(pretrained_weights)

if __name__ == '__main__':
    model = RelTaggerModel(language_model)
    checkpoint = torch.load(_pre_trained_filename, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    test_with_full_match(model, tokenizer, n_ways=10)
