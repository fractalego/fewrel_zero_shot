## Introduction
Code for the paper [Exploring the zero-shot limit of FewRel](https://www.aclweb.org/anthology/2020.coling-main.124). This repository implements a zero-shot relation extractor.

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
connects the entities in the text 
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

| Model                  | 0-shot 5-ways | 0-shot 10-ways |
|------------------------|--------------|----------------|
|(1) Distillbert         |70.1±0.5      | 55.9±0.6       |
|(2) Bert Large          |80.8±0.4      | 69.6±0.5       |
|(3) Distillbert + SQUAD |81.3±0.4      | 70.0±0.2       |
|(4) Bert Large + SQUAD  |86.0±0.6      | 76.2±0.4       |

## Cite as
```bibtex
@inproceedings{cetoli-2020-exploring,
    title = "Exploring the zero-shot limit of {F}ew{R}el",
    author = "Cetoli, Alberto",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.124",
    doi = "10.18653/v1/2020.coling-main.124",
    pages = "1447--1451",
    abstract = "This paper proposes a general purpose relation extractor that uses Wikidata descriptions to represent the relation{'}s surface form. The results are tested on the FewRel 1.0 dataset, which provides an excellent framework for training and evaluating the proposed zero-shot learning system in English. This relation extractor architecture exploits the implicit knowledge of a language model through a question-answering approach.",
}
```

