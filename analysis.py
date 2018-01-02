
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

# clustering kmeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
'''
python analysis.py -n r_37_2_lr1e-1_e200,r_37_2_lr1e-1_best,r_37_2_lr1e-2_e200,r_37_2_lr1e-2_best --prc 
python analysis.py -n r_37_nr1.0E-1_best --km > km_ana_log.txt

'''

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 analysis')
parser.add_argument('-n', '--net', default="net", type=str, help='net name')
parser.add_argument('--debug', '-d', action='store_true', help='debug mode')
parser.add_argument('--prc', action='store_true', help='prec recall curve')
parser.add_argument('--eu', action='store_true', help='Aleatoric uncertainty-entropy plot')
parser.add_argument('--gr', action='store_true', help='plot group')
parser.add_argument('--val', '-v', action='store_true', help='There are val data in training')
parser.add_argument('--km', '-k', action='store_true', help='kmean clustering')
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

def bench_k_means(estimator, name, data, labels):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))
    
def cluster(net,train=False, val=False):
    ## load data
    fn = getMatExt(train=train, val=val)
    # get data & analysis ap, recall
    print(net+'_'+fn+": (k-means clustering)")
    mat = sio.loadmat(os.path.join('cluster', net, fn+'.mat'))
    x_dist = np.array(mat['pred_confidence'])
    y_dist = np.array(mat['ground_truth'])
    x_dist = scale(x_dist)
    n_samples, x_dim = x_dist.shape
    y_dim = y_dist.shape[-1]
    print("y_dim: %d, \t n_samples %d, \t x_dim %d"
      % (y_dim, n_samples, x_dim))
    ## kmeans
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
    bench_k_means(KMeans(init='k-means++', n_clusters=y_dim, n_init=10),
              name=net+"_k-means++", data=x_dist, labels=y_dist)

    bench_k_means(KMeans(init='random', n_clusters=y_dim, n_init=10),
                  name=net+"_random", data=x_dist, labels=y_dist)

    # in this case the seeding of the centers is deterministic, hence we run the
    # kmeans algorithm only once with n_init=1
    pca = PCA(n_components=y_dim).fit(x_dist)
    bench_k_means(KMeans(init=pca.components_, n_clusters=y_dim, n_init=1),
                  name=net+"_PCA-based",
                  data=x_dist, labels=y_dist)
    ## visulize
    reduced_data = PCA(n_components=2).fit_transform(x_dist)
    kmeans = KMeans(init='k-means++', n_clusters=y_dim, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the ' + net + ' result (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    # plt.show()
    plt.savefig(os.path.join(sdir,"plot_"+fn+"_prc.png"), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=fig.dpi)
    plt.close(fig)

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
elif args.km:
    plot_func = cluster
    
    
plots(args.net, plot_func)
# plots(args.net, plot_func, train=True)
if args.val:
    plots(args.net, plot_func, train=True, val=True)
    