'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
# for loss func
import numpy as np
from torch.autograd import Variable

# for coarse lbl
from torch.utils.data import Dataset
from PIL import Image
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

def gcd(a,b):
    """Compute the greatest common divisor of a and b"""
    while b > 0:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Compute the lowest common multiple of a and b"""
    return a * b / gcd(a, b)

def heteroscedastic_uncertainty_loss(criterion, outputs, sig, targets, sna=50):
    loss = Variable(torch.from_numpy(np.array([0.], dtype=np.float)).float().cuda())
    output_avg = Variable(torch.zeros(outputs.data.size()).float().cuda())
    for a in xrange(sna):  # samples mean
        # outputs2 = outputs + sig * Variable(torch.randn(outputs.data.shape).cuda())   # shape for pc only, server use size
        outputs2 = outputs + sig * Variable(torch.randn(outputs.data.size()).cuda())
        output_avg += outputs2
        loss += criterion(outputs2, targets)
    
    loss /= sna
    output_avg /= sna
    return loss, output_avg

def uncertainty_loss(criterion, outputs, sig, targets, sna=50):
    loss = Variable(torch.from_numpy(np.array([0.], dtype=np.float)).float().cuda())
    for a in xrange(sna):  # samples mean
        # outputs2 = outputs + sig * Variable(torch.randn(outputs.data.shape).cuda())   # shape for pc only, server use size
        outputs2 = outputs + sig * Variable(torch.randn(outputs.data.size()).cuda())
        loss += criterion(outputs2, targets)
    
    loss /= sna
    return loss
    
def reassignlbl(fl, cl, sub=-1, reassign=True):
    inlbl = np.asarray(fl)[np.asarray(cl) == sub]
    if not reassign:
        return inlbl.tolist()
    cls = np.unique(inlbl)
    outlbl = inlbl
    for i, c in enumerate(cls):
        outlbl[inlbl == c] = i
    return outlbl.tolist()
    

class ci100dataset(Dataset):
    base_folder = 'cifar-100-python'
    def __init__(self, root, train=True, transform=None, target_transform=None, coarse=False, reassign=True, sub=-1):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.coarse = coarse
        self.sub = sub
        if self.train:
                self.train_labels = []
                
                f = 'train'
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data = entry['data']
                if self.coarse:
                    self.train_labels = entry['coarse_labels']
                else:
                    self.train_labels = entry['fine_labels']
                fo.close()
                
                self.train_data = self.train_data.reshape((50000, 3, 32, 32))
                self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
                if self.sub != -1:
                    self.train_labels = reassignlbl(entry['fine_labels'], entry['coarse_labels'], 
                                                    self.sub, reassign=reassign)
                    self.train_data = self.train_data[np.asarray(entry['coarse_labels']) == self.sub]
        else:
                f = 'test'
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.test_data = entry['data']
                if self.coarse:
                    self.test_labels = entry['coarse_labels']
                else:
                    self.test_labels = entry['fine_labels']
                fo.close()
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
                if self.sub != -1:
                    self.test_labels = reassignlbl(entry['fine_labels'], entry['coarse_labels'], 
                                                    self.sub, reassign=reassign)
                    self.test_data = self.test_data[np.asarray(entry['coarse_labels']) == self.sub]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def softmax(dist):
    return np.exp(dist) / np.array([np.sum(np.exp(dist), axis=1),] *dist.shape[1]).transpose()

def crossEntropy(dist):    
    return (-dist*np.log2(dist)).sum(axis=1)

def softmaxEntropy(dist):
    return crossEntropy(softmax(dist))
    
def rlt2npy(save_dir):
    with open(save_dir + "/losspe.txt", "r") as f:
        losses = f.read()
    with open(save_dir + "/testpe.txt", "r") as f:
        testacc = f.read()
    losses = np.asarray([ float(i) for i in losses.split(" ")])
    testacc = np.asarray([ float(i) for i in testacc.split(" ")])
    dict = {'losses': losses, 
            'testacc': testacc}
    np.save(save_dir+'/train.npy', dict)
    '''
    # load
    d2 = np.load(save_dir+'/train.npy')
    losses = d2.item().get('losses')
    testacc = d2.item().get('testacc')
    '''
    

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
