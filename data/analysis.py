import os
import numpy as np
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine


if __name__ == '__main__':
    LANGUAGES_DICT = {'en': 0, 'fr': 1, 'es': 2, 'it': 3, 'de': 4, 'sk': 5, 'cs': 6}
    def decode_langid(langid):
        for dname, did in LANGUAGES_DICT.items():
            if did == langid:
                return dname

    data = np.load('./samples/lang_samples_132.npz')['data']
    results = data.reshape((7, 250000, 133)).sum(axis=1)

    d_len = len(results)
    distance_matrix = np.ones((d_len, d_len))
    for i in range(d_len):
        for j in range(d_len):
            if i == j:
                continue
            distance_matrix[i, j] = 1 - cosine(results[i], results[j])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(distance_matrix, interpolation='nearest')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + list(LANGUAGES_DICT.keys()))
    ax.set_yticklabels([''] + list(LANGUAGES_DICT.keys()))
    for (i, j), z in np.ndenumerate(distance_matrix):
        ax.text(j, i, '{:0.4f}'.format(z), ha='center', va='center')
    fig.savefig('./out.jpg')
    d = distance_matrix.sum(axis=0)
    d = d / max(d)
    norm = [(float(i) - min(d)) / (max(d) - min(d)) for i in d]
    print(list(zip(norm, list(LANGUAGES_DICT.keys()))))
