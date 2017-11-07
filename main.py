'''Train CIFAR10 with PyTorch.'''
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
# for easydata
from utils import softmaxEntropy
from utils import gcd, lcm
# for loss
from utils import heteroscedastic_uncertainty_loss
# for run-time replace func in class
import types
# for coarse data
from utils import ci100dataset

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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-n', '--net', default="net", type=str, help='net name')
parser.add_argument('--ckptn', '-c', default="best", type=str, help='ckpt name')
parser.add_argument('--dn', default="", type=str, help='data name or ci100')
parser.add_argument('--pn', default="", type=str, help='pretrain name')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
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
parser.add_argument('--rm', action='store_true', help='remove net')
parser.add_argument('--coa', action='store_true', help='cifar100 to coarse lbl')
parser.add_argument('--reas', action='store_true', help='cifar100 to reassign fine lbl')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



if args.dn =="":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif args.dn =="ci100":
    trainset = ci100dataset(root='./data', transform=transform_train, train=True, coarse=args.coa, reassign=args.reas, sub=args.sub)
    testset = ci100dataset(root='./data', transform=transform_test, train=False, coarse=args.coa, reassign=args.reas, sub=args.sub)
else:
    trainset = torch.load('./data/'+args.dn+'.train')
    testset = torch.load('./data/'+args.dn+'.test')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



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
if args.coa:
    net_dir += 'c'
if not args.sub ==-1 and args.reas:
    net_dir += str(args.sub)
if not args.lr == 0.1:
    net_dir += "_lr%.0E"%(args.lr)

print("net: "+net_dir)
if not os.path.isdir('checkpoint/'+net_dir):
    os.makedirs('checkpoint/'+net_dir)


# Model
if args.r3:
    print('==> Resuming from checkpoint to easy new..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+args.pn+'/'+args.ckptn+'_ckpt.t7')
    net = checkpoint['net']
    
    if args.ft:
        net.finetuneLast()

elif args.r2:
    # Load checkpoint.
    print('==> Resuming from checkpoint to 2 lbl..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.pn=="":
        checkpoint = torch.load('./checkpoint/'+args.dn+'/'+args.ckptn+'_ckpt.t7')
    else:
        checkpoint = torch.load('./checkpoint/'+args.pn+'/'+args.ckptn+'_ckpt.t7')
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
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.pn=="":
        checkpoint = torch.load('./checkpoint/'+net_dir+'/'+args.ckptn+'_ckpt.t7')
        best_acc = checkpoint['acc']
    else:
        checkpoint = torch.load('./checkpoint/'+args.pn+'/'+args.ckptn+'_ckpt.t7')
    net = checkpoint['net']
    if not args.ez:
        start_epoch = checkpoint['epoch']
    
    num_cls = max(trainset.train_labels) +1
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
    
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
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
            loss = heteroscedastic_uncertainty_loss(criterion, outputs, sig, targets, sna=args.sna)
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
            loss = heteroscedastic_uncertainty_loss(criterion, outputs, sig, targets, sna=args.sna)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        
        # pdb.set_trace()

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
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
        torch.save(state, './checkpoint/'+net_dir+'/best_ckpt.t7')
        best_acc = acc
    if (epoch+1) % 10 == 0:
        print('Save per 10 epoch and delete last 10')
        try:
            os.remove('./checkpoint/'+net_dir+"/e%03d"%(epoch-9)+'_ckpt.t7')
        except:
            print("No epoch: %03d" % (epoch-9))
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch+1,
        }
        torch.save(state, './checkpoint/'+net_dir+"/e%03d"%(epoch+1)+'_ckpt.t7')


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
    ## load pred
    matpath = os.path.join('mat', net_dir+'_'+args.ckptn, fn+'.mat')
    mat = sio.loadmat(matpath)
    
    ## get score to cut
    dist = mat['pred_confidence']   # get predict
    # dist_sm = np.exp(dist) / np.array([np.sum(np.exp(dist), axis=1),] *dist.shape[1]).transpose()   # softmax
    entropy = softmaxEntropy(dist)   # get cross-entropy to be cut-ts
    
    
    if sca_ts is None:  # gen cut easy ts
        ratio_sort = np.sort(entropy) # small to large = easy to hard
        easy_num = int(dist.shape[0] *0.3)  # take 30% as easy data
        sca_ts = ratio_sort[int(easy_num)]  # choose ts
        
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
        else:
            outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        
        _, predicted = torch.max(outputs.data, 1)
        last_time = time.time()
        test_loss += loss.data[0]
        total += targets.size(0)
        pr = predicted.eq(targets.data).cpu()
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # save data:
        out = outputs.data.cpu().numpy()    # [ 100, 10] no SM
        if args.uc: sig_ = sig.data.cpu().numpy()       # [ 100, 10] no SM
        pred = predicted.cpu().numpy()      # [ 100, ] predict 
        gt = targets.data.cpu().numpy()     # [ 100, ] gt
        runtime += last_time - begin_time   # time(b=100)(sec)
        if batch_idx==0:
            dist = out
            if args.uc: sigs = sig_
            preds = pred
            gts = gt
        else:
            dist = np.concatenate((dist, out), axis=0)
            if args.uc: sigs = np.concatenate((sigs, sig_), axis=0)
            preds = np.concatenate((preds, pred), axis=0)
            gts = np.concatenate((gts, gt), axis=0)
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
                "class_predict: predict class[0:9] \n "
                "predict_right: predict right/wrong[1,0] \n "
                "accuracy: total accuracy[0,100] \n "
                "runtime: runtime on n data in seconds \n "
                "epoch: training epoch[best accu] \n ")
    state = {
        'epoch': start_epoch,
        'pred_confidence': dist, 
        'pred_confidence_sort': dists, 
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
                      'help_str':help_str,})
    if hasattr(trainset, 'val_ind'):
        help_str +="veind: val easy data index in all data \n "
        state.update({'trevind': trainset.val_ind,
                      'help_str':help_str,})
    
    if sigs is not None:
        help_str += "sigma: predict data variance para \n "
        state.update({'sigma': sigs,
                      'help_str':help_str,})
    
    if args.dn =="":
        sdir = os.path.join('mat', net_dir+'_'+args.ckptn)
    else:
        sdir = os.path.join('mat', net_dir+'_'+args.ckptn+'_'+args.dn)
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

def remove_file():
    shutil.rmtree('./checkpoint/'+net_dir, ignore_errors=True)
    shutil.rmtree('./mat/'+net_dir+'_'+args.ckptn, ignore_errors=True)
    try:
        os.remove('./mat/'+net_dir+'_'+args.ckptn+'.zip')
        os.remove('./data/'+args.net+'_2b.train')
        os.remove('./data/'+args.net+'_2b.test')
    except: 
        print("There is not all data exist!")

if args.debug:
    pdb.set_trace()
    # get_zip()
    # test(10)
    # pred2lbl()
    # pred2lbl(train=True)
elif args.rm:
    remove_file()
elif args.zipf:
    get_zip()
elif args.test:
    data_save()
    data_save(train=True)
    if hasattr(trainset, "val_data"):
        data_save(train=True, val=True)
    get_zip()
    print("load epoch: "+ str(start_epoch))
else:
    for epoch in range(start_epoch, start_epoch+args.epoch):
        train(epoch)
        test(epoch)

if args.p2l:
    pred2lbl()
    pred2lbl(train=True)
if args.ed:
    sca_ts = easydata(train=True, val=args.eval)
    easydata(train=False, sca_ts=sca_ts, val=args.eval)
