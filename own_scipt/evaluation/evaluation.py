# coding=utf-8
'''
Created on 2023年1月10日

@author: xiaohejun
'''
import requests
import json
import os
import sys 
import threading
import time
import pickle
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def main(argv):
    if len(argv) < 3:
        print('请输入预测结果文件和真实标签文件')
        return
    pred_logits_file, label_file = argv[1], argv[2]
    with open(pred_logits_file, 'rb') as f:
        pred_logits = pickle.load(f)
    pred_score = F.softmax(torch.Tensor(pred_logits), dim=-1)[:,1]
    pred_label = [1 if item > 0.5 else 0 for item in pred_score]
    with open(label_file, 'r') as f:
        label = []
        for line in f:
            label.append(int(line.strip()))
    accuracy, precision, recall, F1_score, auc = 0, 0, 0, 0, 0
    tp, fp, tn, fn = 0, 0, 0, 0
    for i, j in zip(pred_label, label):
        if i == 1 and j == 1:
            tp += 1
        elif i == 1 and j == 0:
            fp += 1
        elif i == 0 and j == 1:
            fn += 1
        elif i == 0 and j == 0:
            tn += 1
    accuracy = round((tp+tn)/len(label),4)
    precision = round(tp/(tp+fp),4)
    recall = round(tp/(tp+fn),4)
    F1_score = round(2*precision*recall/(precision + recall),4)
    auc = round(roc_auc_score(label, pred_score), 4)
    print('accuracy:\t', accuracy, '\tprecision:\t', precision, '\trecall:\t', recall, '\tF1_score\t', F1_score, '\tauc\t', auc)


if __name__ == '__main__':
    main(sys.argv)




