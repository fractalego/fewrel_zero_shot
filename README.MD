## Introduction
Code for the paper [Exploring the zero-shot limit of FewRel](https://www.aclweb.org/anthology/2020.coling-main.124).

## Dataset
The dataset FewRel 1.0 has been created in the paper 
[ FewRel: A Large-Scale Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation](https://www.aclweb.org/anthology/D18-1514.pdf)
and is available [here](https://github.com/thunlp/FewRel).

## Run the Extractor from the notebook
An example relation extraction is in this [notebook](/notebooks/extractor_examples.ipynb).
The extractor needs a list of candidate relations in English
```python
relations = ['noble title', 'founding date', 'occupation of a person']
extractor = RelationExtractor(model, tokenizer, relations)
```
Then the model ranks the surface forms by the belief that the relation 
connects the entitites in the text 
```python
extractor.rank(text='John Smith received an OBE', head='John Smith', tail='OBE')

[('noble title', 0.9690611883997917),
 ('occupation of a person', 0.0012609362602233887),
 ('founding date', 0.00024014711380004883)]
```

## Training
This repository contains 4 training scripts related to the 4 models in the paper.
```bash
train_bert_large_with_squad.py
train_bert_large_without_squad.py
train_distillbert_with_squad.py
train_distillbert_without_squad.py
```

## Validation
There are also 4 scripts for validation
```bash
test_bert_large_with_squad.py
test_bert_large_without_squad.py
test_distillbert_with_squad.py
test_distillbert_without_squad.py
```

The results as in the paper are

| Model                  | 0-shot5-ways | 0-shot 10-ways |
|------------------------|--------------|----------------|
|(1) Distillbert         |70.1±0.5      | 55.9±0.6       |
|(2) Bert Large          |80.8±0.4      | 69.6±0.5       |
|(3) Distillbert + SQUAD |81.3±0.4      | 70.0±0.2       |
|(4) Bert Large + SQUAD  |86.0±0.6      | 76.2±0.4       |

