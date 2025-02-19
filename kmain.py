#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
'''kmean cluster from pretrain model'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
# import argcomplete

# from models import *
from models import resnet as rn
from utils import progress_bar
from torch.autograd import Variable

import pdb
# pdb.set_trace = lambda: None
import scipy.io as sio
import zipfile
import numpy as np
import time
import shutil
# for save loss
from utils import rlt2npy
from utils import checkRecExist
# for easydata
from utils import softmaxEntropy
from utils import gcd, lcm
# for loss
from utils import heteroscedastic_uncertainty_loss
# for run-time replace func in class
import types
# for coarse data
from utils import ci100dataset
# for noise label
from random import randint
from random import random

'''
# 2lbl
python main.py -n r_37 &&\
python main.py -n r_37 -t -c n_e200 --p2l &&\
python main.py -n r_37_2 --dn r_37_2b --lr 0.01 &&\
python main.py -n r_37_2_lr1e-1 -t --dn r_37_2b -c e200 
# 2 lbl finetune
python main.py -n r_37_2 --r2 --pn r_37 -c n_e200 --ft --dn r_37_2b -e 210 &&\
python main.py -n r_37_2_r2 -t -c e400 --dn r_37_2b

# easy data finetune
python main.py -n r_110 &&\
python main.py -n r_110 -t -c e200 --ed && \
python main.py -n r_37 --r3 --pn r_37 -c n_e200 --dn r_110_ed -e 210 && \
python main.py -n r_37_r3_pr_37_dr_110_ed -t -c e200 --dn r_110_ed

# easy data val
python main.py -n r_110 -t -c e200 --ed --eval && \
python main.py -n r_37 --r3 --pn r_37 -c n_e200 --dn r_110_edv -e 210 && \
python main.py -n r_37_r3_pr_37_dr_110_edv -t -c e200 --dn r_110_edv

# uncertainty
python main.py -n r_37d --uc && \
python main.py -n r_37d3 --uc && \
python main.py -n r_37d --uc --sna 50 -r --pn r_37d3 -c e200 --ez --ccs --uc1d --uc2d && \
python main.py -n r_37d_pr_37d3 --uc -t -c e200

# coarse data
python main.py -n r_37d3 --uc --dn ci100 && \
python main.py -n r_110d3 --uc --dn ci100 && \
python main.py -n r_37d3 --uc --pn r_37d3_dci100 -c e200 --dn ci100 --coa --ez -e 50 && \
python main.py -n r_37d3 --uc --dn ci100 --coa && \
python main.py -n r_37d3 --uc --pn r_37d3_dci100 -c e200 --dn ci100 --sub 0 --reas --ez -e 50 && \
python main.py -n r_110d3 --uc -t -c e200 --dn ci100 && \
python main.py -n r_37d3_pr_37d3 --uc -t -c e050 --dn ci100 --sub 0 --reas

# kmean
python main.py -n r_37d3 --uc --kmean -t -c e050



matlab:
matlab -nosplash -nodesktop
    $ load('fn.mat')
    load('mat/r_37_n_e200/test.mat')
    who $ to watch variable
'''
'''
# check code args:
from inspect import getargspec as ga
print(ga(net.forward))
# check source code:
import inspect
lines = inspect.getsourcelines(net.forward)
print("".join(lines[0]))
'''

def getdir(root, prefix):
    dirname = [name for name in os.listdir(root) if os.path.isdir(root+'/'+name)]
    return ( name for name in dirname if name.startswith(prefix) )


def getfile(root, prefix):
    dirname = [os.path.splitext(name)[0] for name in os.listdir(root) if os.path.isfile(root+'/'+name)]
    dirname = list(set(dirname))
    return ( name for name in dirname if name.startswith(prefix) )


def ckpt_nets(prefix, parsed_args, **kwargs):
    return getdir(parsed_args.checkpointdir, prefix)

def mat_nets(prefix, parsed_args, **kwargs):
    return getdir('mat', prefix)

def data_name(prefix, parsed_args, **kwargs):
    return getfile('data', prefix)
    
def ckpt_epoch(prefix, parsed_args, **kwargs):
    if parsed_args.pn=="":
        return getfile(parsed_args.checkpointdir+'/'+parsed_args.net, prefix)
    else:
        return getfile(parsed_args.checkpointdir+'/'+parsed_args.pn, prefix)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('-n', '--net', type=str, help='net name', choices = ckpt_nets())
parser.add_argument('-n', '--net', type=str, help='net name').completer = ckpt_nets
parser.add_argument('--ckptn', '-c', default="best", type=str, help='ckpt name').completer = ckpt_epoch
parser.add_argument('--dn', default="", type=str, help='data name or ci100').completer = data_name
parser.add_argument('--pn', default="", type=str, help='pretrain name').completer = ckpt_nets
parser.add_argument('--checkpointdir', default="./checkpoint", type=str, help='checkpoint dir')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--nr', default=0., type=float, help='noise label rate')
parser.add_argument('--epoch', '-e', default=200, type=int, help='next training epoch')
parser.add_argument('--sna', '-a', default=50, type=int, help='sample num for Aleatoric Uncertainty')
parser.add_argument('--sub', default=-1, type=int, help='sub-set of coarse lbl ind in cifar100, int [0-19]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true', help='test checkpoint')
parser.add_argument('--zipf', '-z', action='store_true', help='zip the test')
parser.add_argument('--debug', '-d', action='store_true', help='debug mode')
parser.add_argument('--p2l', action='store_true', help='pred mat to lbl')
parser.add_argument('--ed', action='store_true', help='pred mat to easy data')
parser.add_argument('--eval', action='store_true', help='easy train data split val')
parser.add_argument('--r2', action='store_true', help='resume to 2 lbl')
parser.add_argument('--r3', action='store_true', help='resume to easy')
parser.add_argument('--uc', action='store_true', help='uncertainty training')
parser.add_argument('--ez', action='store_true', help='epoch to zero')
parser.add_argument('--ccs', action='store_true', help='change cd setting')
parser.add_argument('--uc1d', action='store_true', help='c1d')
parser.add_argument('--uc2d', action='store_true', help='c2d')
parser.add_argument('--ft', action='store_true', help='finetune only last')
parser.add_argument('--coa', action='store_true', help='cifar100 to coarse lbl')
parser.add_argument('--reas', action='store_true', help='cifar100 to reassign fine lbl')
parser.add_argument('--kmean', action='store_true', help='k-mean clustering')
# argcomplete.autocomplete(parser)
args = parser.parse_args()

# save dir
net_dir = args.net
if args.r3:
    net_dir += '_r3'
if args.r2:
    net_dir += '_r2'
if args.ft:
    net_dir += '_ft'
if not args.pn =="":
    net_dir += '_p' + args.pn
if not args.dn =="" and not args.test:
    net_dir += '_d' + args.dn
if args.coa and not args.test:
    net_dir += 'c'
if not args.sub ==-1 and args.reas and not args.test:
    net_dir += str(args.sub)
if not args.lr == 0.1:
    net_dir += "_lr%.0E"%(args.lr)
if not args.nr == 0.:
    net_dir += "_nr%.1E"%(args.nr)

print("net: "+net_dir)
checkpointdir = args.checkpointdir+'/'+net_dir
if not os.path.isdir(checkpointdir):
    os.makedirs(checkpointdir)
file_name = os.path.join(checkpointdir, 'opt.txt')
with open(file_name, 'wt') as opt_file:
    opt_file.write('------------ Options -------------\n')
    for k, v in sorted(vars(args).items()):
        opt_file.write('%s: %s\n' % (str(k), str(v)))
    opt_file.write('-------------- End ----------------\n')

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.test:
    transform_train = transform_test
else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

if args.nr != 0.:
    if args.dn =="ci100":
        high = 100
    else:
        high = 10
    target_transform = lambda t: randint(0, high-1) if args.nr > random() else t
    '''
    target_transform = transforms.Compose([
            transforms.Lambda(lambda t: rand_tar(t)),
        ])
    '''
else:
    target_transform = None


if args.dn =="":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train, target_transform=target_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif args.dn =="ci100":
    trainset = ci100dataset(root='./data', transform=transform_train, target_transform=target_transform, train=True, coarse=args.coa, reassign=args.reas, sub=args.sub)
    testset = ci100dataset(root='./data', transform=transform_test, train=False, coarse=args.coa, reassign=args.reas, sub=args.sub)
else:
    trainset = torch.load('./data/'+args.dn+'.train')
    testset = torch.load('./data/'+args.dn+'.test')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




# Model
if args.r3:
    print('==> Resuming from checkpoint to easy new..')
    assert os.path.isdir(checkpointdir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.checkpointdir+'/'+args.pn+'/'+args.ckptn+'_ckpt.t7')
    net = checkpoint['net']
    
    if args.ft:
        net.finetuneLast()

elif args.r2:
    # Load checkpoint.
    print('==> Resuming from checkpoint to 2 lbl..')
    assert os.path.isdir(args.checkpointdir), 'Error: no checkpoint directory found!'
    if args.pn=="":
        checkpoint = torch.load(args.checkpointdir+'/'+args.dn+'/'+args.ckptn+'_ckpt.t7')
    else:
        checkpoint = torch.load(args.checkpointdir+'/'+args.pn+'/'+args.ckptn+'_ckpt.t7')
    net = checkpoint['net']
    start_epoch = checkpoint['epoch']
    # replace para
    if args.net =="r_37_2":
        net.linear = nn.Linear(4096, 2)
    
    if args.ft:
        net.finetuneLast()
    
elif args.resume or args.test or args.pn!="":
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.checkpointdir), 'Error: no checkpoint directory found!'
    if args.pn=="":
        checkpoint = torch.load(args.checkpointdir+'/'+net_dir+'/'+args.ckptn+'_ckpt.t7')
        best_acc = checkpoint['acc']
    else:
        checkpoint = torch.load(args.checkpointdir+'/'+args.pn+'/'+args.ckptn+'_ckpt.t7')
    net = checkpoint['net']
    if not args.ez:
        start_epoch = checkpoint['epoch']
    
    num_cls = len(np.unique(trainset.train_labels))
    if (not args.test) and net.linear.out_features != num_cls:
        net.genNewLin(num_cls)
    
    if args.ccs:
        if not hasattr(net, "setdrop"): # run-time replace func in class
            print("add net.setdrop")
            def setdrop(self, c1d, c2d):
                self.c1d = c1d
                self.c2d = c2d
            net.setdrop = types.MethodType(setdrop, net)
        print("set net.setdrop")
        net.setdrop(args.uc1d, args.uc2d)
    if args.kmean:
        def lastFeature(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            if self.num_layers >0:
                out = self.layer1(out)
            if self.num_layers >1:
                out = self.layer2(out)
            if self.num_layers >2:
                out = self.layer3(out)
            if self.num_layers >3:
                out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            return out
        net.lastFeature = types.MethodType(lastFeature, net)
    if args.ft:
        net.finetuneLast()

else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    if hasattr(rn, args.net):
        net = getattr(rn, args.net)()
    else:
        print('There is no ' + args.net)
        exit()
    if args.dn =="ci100":
        net.genNewLin(max(trainset.train_labels) +1)


# check net is uc-net
if args.uc:
    args.uc = hasattr(net, 'sig')
elif hasattr(net, 'sig'):
    print("net has sig, but no \"--uc\"")
    
if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


if not hasattr(trainset, 'weights'):
    criterion = nn.CrossEntropyLoss()
else:
    weights = torch.from_numpy(trainset.weights)
    weights = weights.type(torch.FloatTensor)
    if use_cuda:
        weights = weights.cuda()
    criterion = nn.CrossEntropyLoss(weight=weights)
    
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# branchy case:optim=Adam, lr=0.1, momentum=0.9, weight_decay=0.0001, alpha=0.001
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    
    
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        if args.uc:
            outputs, sig = net(inputs)
            loss,_ = heteroscedastic_uncertainty_loss(criterion, outputs, sig, targets, sna=args.sna)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    # save loss
    lossfn = os.path.join(checkpointdir, "losspe.txt")
    with open(lossfn, "a") as f:
        f.write(str(train_loss/(batch_idx+1))+" ")


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        if args.uc:
            outputs, sig = net(inputs)
            loss, output_avg = heteroscedastic_uncertainty_loss(criterion, outputs, sig, targets, sna=args.sna)
            _, predicted = torch.max(output_avg.data, 1)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
        
        # pdb.set_trace()

        test_loss += loss.data[0]
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch+1,
        }
        torch.save(state, checkpointdir+'/best_ckpt.t7')
        best_acc = acc
    if (epoch+1) % 10 == 0:
        print('Save per 10 epoch and delete last 10')
        try:
            os.remove(checkpointdir+"/e%03d"%(epoch-9)+'_ckpt.t7')
        except:
            print("No epoch: %03d" % (epoch-9))
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch+1,
        }
        torch.save(state, checkpointdir+"/e%03d"%(epoch+1)+'_ckpt.t7')
    
    # save testing accu
    teaccfn = os.path.join(checkpointdir, "testpe.txt")
    with open(teaccfn, "a") as f:
        f.write(str(acc)+" ")

def pred2lbl(train=False):
    if train:
        fn = "train"
    else:
        fn = "test"
    # load pred
    matpath = os.path.join('mat', net_dir+'_'+args.ckptn, fn+'.mat')
    mat = sio.loadmat(matpath)
    # change to lbl
    pr = mat['predict_right']   # right: 1; wrong: 0
    pr = np.squeeze(pr)
    # weight cls
    weights = []
    for i in range(min(pr), max(pr)+1):
        weights.append((len(pr)+0.)/sum(pr==i))
    
    weights = np.array(weights)
    weights = weights * len(weights)/ sum(weights)  # 0918: nor sigma = num of cls
    
    pr = pr.tolist()
    # conbime with data
    if train:
        dataset = trainset
        dataset.train_labels = pr
    else:
        dataset = testset
        dataset.test_labels = pr
    dataset.weights = weights
    # save net datasets
    torch.save(dataset, './data/'+args.net+'_2b.'+fn)
    
def easydata(train=False, sca_ts=None, val=False):
    if train:
        fn = "train"
        dataset = trainset
    else:
        fn = "test"
        dataset = testset
    if val:
        savefn = './data/'+net_dir+'_edv.'+fn
    else:
        savefn = './data/'+net_dir+'_ed.'+fn
    
    if args.dn =="":
        matpath = os.path.join('mat', net_dir+'_'+args.ckptn, fn+'.mat')
    else:
        matpath = os.path.join('mat', net_dir+'_'+args.ckptn+'_'+args.dn, fn+'.mat')
    ## load pred
    mat = sio.loadmat(matpath)
    
    ## get score to cut
    dist = mat['pred_confidence']   # get predict
    # dist_sm = np.exp(dist) / np.array([np.sum(np.exp(dist), axis=1),] *dist.shape[1]).transpose()   # softmax
    entropy = softmaxEntropy(dist)   # get cross-entropy to be cut-ts
    
    
    if sca_ts is None:  # gen cut easy ts
        ratio_sort = np.sort(entropy) # small to large = easy to hard
        easy_num = int(dist.shape[0] *0.3)  # take 30% as easy data
        sca_ts = ratio_sort[int(easy_num)]  # choose ts
        print("sca_ts: %3e" % sca_ts)
        
    easy_ind = np.where(entropy < sca_ts)
    easy_ind = np.asarray(easy_ind)
    easy_ind = np.squeeze(easy_ind)
    
    ## get val data
    val_ind = None
    easy_num = easy_ind.shape[0]
    if train and val:
        val_ind = np.arange(easy_num)
        np.random.shuffle(val_ind)
        val_ind = val_ind[:int((easy_num+0.) / 5)]  # 1/5 of easy
        val_ind = np.sort(val_ind)
        dataset.val_data = dataset.train_data[easy_ind[val_ind]]
        dataset.val_labels = np.asarray(dataset.train_labels)[easy_ind[val_ind]].tolist()
        dataset.val_ind = easy_ind[val_ind]
        easy_ind = np.delete(easy_ind, val_ind)
        easy_num = easy_ind.shape[0]
    
    ## split data
    pr = None
    if hasattr(dataset, 'train_data'):
        dataset.train_data = dataset.train_data[easy_ind]
        dataset.train_labels = np.asarray(dataset.train_labels)[easy_ind].tolist()
        pr = np.asarray(dataset.train_labels)
    else:
        dataset.test_data = dataset.test_data[easy_ind]
        dataset.test_labels = np.asarray(dataset.test_labels)[easy_ind].tolist()
        pr = np.asarray(dataset.test_labels)
        
    ## get ind close ts
    ratio_sort_ind = np.argsort(entropy)
    ratio_sort_ind = np.asarray(ratio_sort_ind)
    ratio_sort_ind = np.squeeze(ratio_sort_ind)
    easy_close_ind = ratio_sort_ind[easy_num -10:easy_num]   # close last to be hard
    hard_close_ind = ratio_sort_ind[easy_num +1:easy_num +11]   # close first to be easy
    
    ## weight cls
    weights = np.zeros( max(pr)+1)
    for i in range(min(pr), max(pr)+1):
        weights[i] = sum(pr==i) # get num per cls
        if i ==0: pass
        elif i ==1:               # get lcm to cal wei
            w_lcm = lcm(weights[0], weights[1])
        else:
            w_lcm = lcm(w_lcm, weights[i])
    
    weights = w_lcm / weights
    weights = weights * len(weights)/ sum(weights)  # 0918: nor sigma = num of cls
    
    dataset.weights = weights
    dataset.easy_close_ind = easy_close_ind
    dataset.hard_close_ind = hard_close_ind
    dataset.sca_ts = sca_ts
    dataset.easy_ind = easy_ind
    # save net datasets
    torch.save(dataset, savefn)
    return sca_ts

def data_save(train=False, val=False):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    runtime = 0.
    # load data
    if train:
        fn = "train"
        if val:
            fn = "trvali"
            trainset.train_data = trainset.val_data
            trainset.train_labels = trainset.val_labels
        loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)
    else:
        fn = "test"
        loader = testloader
    sigs = None
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        last_time = time.time()
        begin_time = last_time
        if args.uc:
            outputs, sig = net(inputs)
            loss_u, output_avg = heteroscedastic_uncertainty_loss(criterion, outputs, sig, targets, sna=args.sna)
            _, predicted = torch.max(outputs.data, 1)
            _, pred_u = torch.max(output_avg.data, 1)
            correct += pred_u.eq(targets.data).cpu().sum()
        else:
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum()
            
        
        loss = criterion(outputs, targets)
        last_time = time.time()
        test_loss += loss.data[0]
        total += targets.size(0)

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # save data:
        out = outputs.data.cpu().numpy()    # [ 100, 10] no SM
        ent = softmaxEntropy(out)
        if args.uc: 
            sig_ = sig.data.cpu().numpy()       # [ 100, 10] no SM
            pred_u_ = pred_u.cpu().numpy()      # [ 100, ] predict 
            output_avg_ = output_avg.data.cpu().numpy()
            ent_un = softmaxEntropy(output_avg_)
        pred = predicted.cpu().numpy()      # [ 100, ] predict 
        gt = targets.data.cpu().numpy()     # [ 100, ] gt
        runtime += last_time - begin_time   # time(b=100)(sec)
        if batch_idx==0:
            dist = out
            preds = pred
            gts = gt
            ents = ent
            if args.uc: 
                dist_avgs = output_avg_
                sigs = sig_
                pred_us = pred_u_
                ent_uns = ent_un
        else:
            dist = np.concatenate((dist, out), axis=0)
            preds = np.concatenate((preds, pred), axis=0)
            gts = np.concatenate((gts, gt), axis=0)
            ents = np.concatenate((ents, ent), axis=0)
            if args.uc: 
                dist_avgs = np.concatenate((dist_avgs, output_avg_), axis=0)
                sigs = np.concatenate((sigs, sig_), axis=0)
                pred_us = np.concatenate((pred_us, pred_u_), axis=0)
                ent_uns = np.concatenate((ent_uns, ent_un), axis=0)
    # Save checkpoint.
    acc = 100.*correct/total
    preds = preds.squeeze()
    prbool = np.equal(preds, gts)
    pright = np.ones(preds.shape, dtype=np.int8)
    pright[prbool==False] = 0.
    print('Saving data..')
    dists = np.sort(dist, axis=-1)
    help_str = ("pred_confidence: distribution of predicted confidence, array[n, d] \n "
                "n: num of data, d: num of class \n "
                "*_sort: sorted by d(small to large) \n "
                "class_predict: predict class[0:c] \n "
                "cross_entropy: cross-entropy of pred_confidence, array[n, 1]  \n "
                "predict_right: predict right/wrong[1,0] \n "
                "accuracy: total accuracy[0,100] \n "
                "runtime: runtime on n data in seconds \n "
                "epoch: training epoch[best accu] \n ")
    state = {
        'epoch': start_epoch,
        'pred_confidence': dist, 
        'pred_confidence_sort': dists, 
        'cross_entropy': ents, 
        'ground_truth': gts, 
        'class_predict': preds, 
        'predict_right': pright, 
        'accuracy': acc, 
        'runtime': runtime,
        'help_str':help_str,
    }
    if hasattr(trainset, 'easy_close_ind'):
        help_str +=("treid: train easy data id which is close to ts \n "
                    "trhid: train hard data id which is close to ts \n "
                    "teeid: test easy data id which is close to ts \n "
                    "tehid: test hard data id which is close to ts \n "
                    "ent_ts: the ts splitting data, generating from cross-entropy \n "
                    "treind: easy train data index in all data \n "
                    "teeind: easy test data index in all data \n ")
        state.update({'treid': trainset.easy_close_ind,
                      'trhid': trainset.hard_close_ind, 
                      'teeid': testset.easy_close_ind, 
                      'tehid': testset.hard_close_ind, 
                      'ent_ts': trainset.sca_ts,
                      'treind': trainset.easy_ind,
                      'teeind': testset.easy_ind,
                      'help_str': help_str,})
    if hasattr(trainset, 'val_ind'):
        help_str += "veind: val easy data index in all data \n "
        state.update({'trevind': trainset.val_ind,
                      'help_str': help_str,})
    
    if sigs is not None:
        help_str +=("sigma: predict data variance para \n "
                    "pred_confidence_avg: Gaussian sample average predicted confidance, array[n, d] \n "
                    "class_predict_avg: Gaussian sample average predict class[0:c] \n "
                    "cross_entropy_avg: average cross-entropy of pred_confidence, array[n, 1]  \n ")
        state.update({'sigma': sigs,
                      'pred_confidence_avg': dist_avgs,
                      'class_predict_avg': pred_us,
                      'cross_entropy_avg': ent_uns,
                      'help_str': help_str,})
    
    if args.dn =="":
        sdir = os.path.join('mat', net_dir+'_'+args.ckptn)
    else:
        sdir = os.path.join('mat', net_dir+'_'+args.ckptn+'_'+args.dn)
    if args.coa:
        sdir+='c'
    if not os.path.isdir(sdir):
        os.makedirs(sdir)
    sio.savemat(sdir+'/'+fn+'.mat', state, do_compression=True)
def km_cluster_data_save(train=False, val=False):
    net.eval()
    test_loss = 0
    total = 0
    runtime = 0.
    # load data
    if train:
        fn = "train"
        if val:
            fn = "trvali"
            trainset.train_data = trainset.val_data
            trainset.train_labels = trainset.val_labels
        loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)
    else:
        fn = "test"
        loader = testloader
    # inference data
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        last_time = time.time()
        begin_time = last_time
        outputs = net.lastFeature(inputs)


        last_time = time.time()
        total += targets.size(0)

        progress_bar(batch_idx, len(loader), 'Loss:')
        # extract data:
        out = outputs.data.cpu().numpy()    # [ 100, 10] no SM
        gt = targets.data.cpu().numpy()     # [ 100, ] gt
        runtime += last_time - begin_time   # time(b=100)(sec)
        if batch_idx == 0:
            dist = out
            gts = gt
        else:
            dist = np.concatenate((dist, out), axis=0)
            gts = np.concatenate((gts, gt), axis=0)
    # Save checkpoint.
    print('Saving data..')
    help_str = ("pred_confidence: distribution of predicted confidence, array[n, d] \n "
                "n: num of data, d: num of class \n "
                "runtime: runtime on n data in seconds \n "
                "epoch: training epoch[best accu] \n ")
    state = {
        'epoch': start_epoch,
        'pred_confidence': dist,
        'ground_truth': gts,
        'runtime': runtime,
        'help_str':help_str,
    }
    if hasattr(trainset, 'val_ind'):
        help_str += "veind: val easy data index in all data \n "
        state.update({'trevind': trainset.val_ind,
                      'help_str': help_str,})

    sdir = os.path.join('cluster', net_dir+'_'+args.ckptn)
    if args.dn !="":
        sdir += '_'+args.dn
    if args.coa:
        sdir += 'c'
    if not os.path.isdir(sdir):
        os.makedirs(sdir)
    sio.savemat(sdir+'/'+fn+'.mat', state, do_compression=True)

def get_zip():
    if args.dn =="":
        fn = net_dir +'_'+args.ckptn
    else:
        fn = net_dir +'_'+args.ckptn+'_'+args.dn
    zf = zipfile.ZipFile('./mat/'+fn + '.zip', mode='w', compression = zipfile.ZIP_DEFLATED)
    for root, folders, files in os.walk("./mat"):
        if root == './mat/'+fn :
            print("root: " + root)
            for sfile in files:
                if sfile.find('.mat') > -1:
                    aFile = os.path.join(root, sfile)
                    #   print aFile
                    zf.write(aFile)


if args.debug:
    pdb.set_trace()
    # get_zip()
    # test(10)
    # pred2lbl()
    # pred2lbl(train=True)
elif args.zipf:
    get_zip()
elif args.kmean:
    print("kmean:")
    km_cluster_data_save()
    km_cluster_data_save(train=True)
elif args.test:
    data_save()
    data_save(train=True)
    if hasattr(trainset, "val_data"):
        data_save(train=True, val=True)
    get_zip()
    print("load epoch: "+ str(start_epoch))
else:
    # checkRecExist(checkpointdir)
    for epoch in range(start_epoch, start_epoch+args.epoch):
        train(epoch)
        test(epoch)
        
    rlt2npy(checkpointdir)

if args.p2l:
    pred2lbl()
    pred2lbl(train=True)
if args.ed:
    sca_ts = easydata(train=True, val=args.eval)
    easydata(train=False, sca_ts=sca_ts, val=args.eval)
