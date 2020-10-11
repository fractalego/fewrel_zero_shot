import random
import time

import torch
from tqdm import tqdm


def train(train_model, batches, optimizer, criterion):
    total_loss = 0.
    for i, batch in tqdm(enumerate(batches), total=len(batches)):
        inputs, subj_start_targets, subj_end_targets, obj_start_targets, obj_end_targets, lengths = \
            batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        optimizer.zero_grad()
        subj_start, subj_end, obj_start, obj_end = train_model(inputs.cuda(), lengths)
        loss1 = criterion(subj_start, subj_start_targets.cuda().float())
        loss2 = criterion(subj_end, subj_end_targets.cuda().float())
        loss3 = criterion(obj_start, obj_start_targets.cuda().float())
        loss4 = criterion(obj_end, obj_end_targets.cuda().float())
        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        time.sleep(0.05)

    return total_loss


def select_sample(lst1, lst2, ratio):
    lst = list(zip(lst1, lst2))
    random.shuffle(lst)
    lst = lst[:int(len(lst) * ratio)]
    return [item[0] for item in lst], [item[1] for item in lst]
