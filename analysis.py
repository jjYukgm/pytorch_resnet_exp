
import os
import argparse

import pdb
# pdb.set_trace = lambda: None
import scipy.io as sio
import zipfile
import numpy as np
import shutil

# plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
# pr curve
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from utils import softmaxEntropy
'''
python analysis.py -n r_37_2_lr1e-1_e200,r_37_2_lr1e-1_best,r_37_2_lr1e-2_e200,r_37_2_lr1e-2_best --prc 
'''

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 analysis')
parser.add_argument('-n', '--net', default="net", type=str, help='net name')
parser.add_argument('--debug', '-d', action='store_true', help='debug mode')
parser.add_argument('--prc', action='store_true', help='prec recall curve')
parser.add_argument('--eu', action='store_true', help='Aleatoric uncertainty-entropy plot')
parser.add_argument('--gr', action='store_true', help='plot group')
parser.add_argument('--val', '-v', action='store_true', help='There are val data in training')
args = parser.parse_args()

# for analysis
def getMatExt(train=False, val=False):
    if train:
        fn = "train"
        if val:
            fn = "trvali"
    else:
        fn = "test"
    return fn

def plots(nets, plot_func, train=False, val=False):
    nets = nets.split(",")
    for n in nets:
        plot_func(n, train=train, val=val)

def pr_curve(net,train=False, val=False):
    ## load pred
    fn = getMatExt(train=train, val=val)
    # get data & analysis ap, recall
    print(net+'_'+fn+": (recall, prec)")
    mat = sio.loadmat(os.path.join('mat', net, fn+'.mat'))
    y_score = np.array(mat['pred_confidence'])
    num_cls = y_score.shape[-1]
    lb = preprocessing.LabelBinarizer()
    lb.fit(range(0, num_cls))
    gt_bi = lb.transform(np.array(mat['ground_truth']).squeeze())
    if num_cls==2:
        gt_bi = np.hstack((gt_bi, 1 - gt_bi))
    
    ## For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(num_cls):
        precision[i], recall[i], _ = precision_recall_curve(gt_bi[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(gt_bi[:, i], y_score[:, i])

    ## A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(gt_bi.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(gt_bi, y_score, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
    
    ## plot
    lines = []
    labels = []

    for i in range(num_cls):
        l, = plt.plot(recall[i], precision[i], lw=2)
        lines.append(l)
        labels.append('class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.75)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve to multi-class')
    lgd = plt.legend(lines, labels, loc=(0, -1.0), prop=dict(size=14))
    ## save
    sdir = os.path.join('plot',net)
    if not os.path.isdir(sdir):
        os.makedirs(sdir)
    plt.tight_layout()
    plt.savefig(os.path.join(sdir,"plot_"+fn+"_prc.png"), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=fig.dpi)
    plt.close(fig)
 

def entropyUncertainty(net,train=False, val=False):
    ## load pred
    fn = getMatExt(train=train, val=val)
    # get data & analysis ap, recall
    print(net+'_'+fn+": (entropy, Aleatoric Uncertainty)")
    mat = sio.loadmat(os.path.join('mat', net, fn+'.mat'))
    if not 'sigma' in mat:
        print('net: %s/%s has no pred sigma' % (net, fn))
        return

    y_score = np.array(mat['pred_confidence'])
    ent = softmaxEntropy(y_score)
    ent = np.squeeze(ent)
    unc = np.array(mat['sigma'])
    unc = np.absolute(unc)
    unc = np.mean(unc, axis=1)
    unc = np.squeeze(unc)
    
    ## plot
    # lines = []
    # labels = []

    plt.plot(ent, unc, ',')
    # lines.append(l)
    # labels.append('net: ' + net)
    fig = plt.gcf()
    # fig.subplots_adjust(bottom=0.75)
    plt.xlim([min(ent), max(ent)])
    plt.ylim([min(unc), max(unc)])
    plt.xlabel('Cross-Entropy')
    plt.ylabel('Aleatoric Uncertainty')
    plt.title('Two Threshold Distribution Compare')
    # lgd = plt.legend(lines, labels, loc=(0, -1.0), prop=dict(size=14))
    ## save
    sdir = os.path.join('plot',net)
    if not os.path.isdir(sdir):
        os.makedirs(sdir)
    plt.tight_layout()
    plt.savefig(os.path.join(sdir,"plot_"+fn+"_eu.png"), bbox_inches='tight', dpi=fig.dpi)
    plt.close(fig)
 
def remove_file():
    if args.net.find(',') > -1:
        nets = nets.split(",")
        for n in nets:
            shutil.rmtree(os.path.join('plot', n), ignore_errors=True)
    else:
        shutil.rmtree(os.path.join('plot', args.net), ignore_errors=True)

plt.ioff() 

# confirm dir exist
if not os.path.isdir('plot'):
    os.mkdir('plot')
if args.debug:
    # debug use
    pdb.set_trace()
    pr_curve(args.net)
    # pass 
if args.prc:
    plot_func = pr_curve
elif args.eu:
    plot_func = entropyUncertainty
    
    
plots(args.net, plot_func)
plots(args.net, plot_func, train=True)
if args.val:
    plots(args.net, plot_func, train=True, val=True)
    