# -*- coding: utf-8 -*-
"""

@author: dongdong1994
"""

import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#中文显示
def getChineseFont():
    return FontProperties(fname='/usr/share/fonts/SimHei.ttf')

def plot_with_lables(X, Y, filename="images/tsne.png"):
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(Y):
        x, y = X[i, :]
        plt.scatter(x, y)
        plt.annotate(label, 
                     xy=(x, y), 
                     xytext=(5,2), 
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     fontproperties=getChineseFont())
    plt.savefig(filename)

if __name__ == "__main__":
    dec_vec_model = gensim.models.Word2Vec.load(r'model/decoder_vector.m')
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    dec_useful_words = list(dec_vec_model.wv.vocab.keys())
    
    plot_only=200
    
    X = []
    Y_LABELS = []
    counter = 0
    for elem in dec_useful_words:
        X.append(dec_vec_model.wv[elem])
        #byte str互相钻换
        Y_LABELS.append(elem.encode("utf8").decode('utf8'))
        if counter > plot_only:
            break
        counter += 1
    X_TRANS = tsne.fit_transform(X)
    
    plot_with_lables(X_TRANS, Y_LABELS)