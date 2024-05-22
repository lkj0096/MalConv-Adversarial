# -*- coding: utf-8 -*-
import os
import time
import sys
import yaml
import numpy as np
import pandas as pd
from src.util import ExeDataset, write_pred
from src.model import MalConv
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

# Load config file for experiment
# from torch import load

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath((__file__))))
    config_path = 'config/example.yaml'
    seed = int(123)
    conf = yaml.safe_load(open(config_path, 'r'))

    use_gpu = conf['use_gpu']
    use_cpu = conf['use_cpu']
    exp_name = conf['exp_name'] + '_sd_' + str(seed)

    valid_data_path = conf['valid_data_path']
    valid_label_path = conf['valid_label_path']

    checkpoint_dir = conf['checkpoint_dir']
    chkpt_acc_path = checkpoint_dir + exp_name + '.model'

    val_label_table = pd.read_csv(valid_label_path, header=None, index_col=0)
    val_label_table.index = val_label_table.index.str.upper()
    val_label_table = val_label_table.rename(columns={1: 'ground_truth'})

    val_table = val_label_table.groupby(level=0).last()
    del val_label_table

    validloader = DataLoader(ExeDataset(list(val_table.index), valid_data_path, list(val_table.ground_truth), 2000000),
                             batch_size=1, shuffle=False, num_workers=use_cpu)

malconv = torch.load('checkpoint/example_sd_123.model')

    history = {}
    history['val_loss'] = []
    history['val_acc'] = []
    history['val_pred'] = []
    bce_loss = nn.BCEWithLogitsLoss()
    total=0
    evade = 0
    changes=[]
    for _, val_batch_data in enumerate(validloader):
        total+=1
        cur_batch_size = val_batch_data[0].size(0)
        print("cur batch size:", cur_batch_size)

        exe_input = val_batch_data[0].cuda() if use_gpu else val_batch_data[0]
        data = exe_input[0].cpu().numpy()
        length = data[-1]

        data = data[:length]
        data = np.concatenate([data, np.random.randint(1, 256, 2000000 - length)])
        data[-1] = 0
        label = val_batch_data[1].cuda() if use_gpu else val_batch_data[1]
        label = Variable(label.float(), requires_grad=False)
        label = Variable(torch.from_numpy(np.array([[0]])).float(), requires_grad=False)

        embed = malconv.embed
        sigmoid = nn.Sigmoid().cuda()
        for t in tqdm(range(20)):                                        #original: 20
            exe_input = torch.from_numpy(np.array([data]))
            exe_input = Variable(exe_input.long(), requires_grad=False).cuda()
            pred = malconv(exe_input)
            prob = sigmoid(pred).cpu().data.numpy()[0][0]
            print("prob: ", prob)
            if prob < 0.5:
                print("prob<0.5,success.")
                evade+=1
                print("evading rate:",evade/float(total))
                break

            loss = bce_loss(pred, label)
            loss.backward()
            w = malconv.embed_x.grad[0].data
            z = malconv.embed_x.data[0]
            for j in tqdm(range(length, length + 100000)):           # Original: 100000
                if j % 20000 == 0:
                    exe_input = torch.from_numpy(np.array([data]))
                    exe_input = Variable(exe_input.long(), requires_grad=False).cuda()
                    pred = malconv(exe_input)
                    prob = sigmoid(pred).cpu().data.numpy()[0][0]
                    print("prob: ", prob)
                    if prob < 0.5:
                        break
                    print("change " + str(j) + "th byte")
                try:
                    min_index = -1
                    min_di = 100000
                    wj = -w[j:j + 1, :]
                    nj = wj / torch.norm(wj, 2)
                    zj = z[j:j + 1, :]
                    for i in range(1, 256):
                        mi = embed(Variable(torch.from_numpy(np.array([i]))).cuda()).data
                        si = torch.matmul((nj), torch.t(mi - zj))
                        di = torch.norm(mi - (zj + si * nj))
                        si = si.cpu().numpy()
                        if si > 0 and di < min_di:
                            min_di = di
                            min_index = i
                    if min_index != -1:
                        data[j] = min_index
                        changes.append(min_index)
                except Exception as e:
                    print(e)
                    continue
            print("finish ", t)
    changes=np.array(changes)
    print(changes)
    np.save("changes.npy",changes)
