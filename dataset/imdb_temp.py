# -*- coding: utf-8 -*-

import numpy as np
import json
from preprocessing.sequence import _remove_long_seq

def load_data(imfile = 'E:\pest1\imdb\imdb.npz', num_words=None, skip_top=0,
              maxlen=None, seed=113,
              start_char=1, oov_char=2, index_from=3, **kwargs):
    npzfile = np.load(imfile)
    npzfile.files
    x_test = npzfile['x_test']
    x_train = npzfile['x_train']
    labels_train = npzfile['y_train']
    labels_test = npzfile['y_test']
    
    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(labels_train)
    
    np.random.seed(seed * 2)
    np.random.shuffle(x_test)
    np.random.seed(seed * 2)
    np.random.shuffle(labels_test)
    
    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])
    
    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]
        
    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                             'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])
        
    if oov_char is not None:
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
    else:
        xs = [[w for w in x if (skip_top <= w < num_words)] for x in xs]
        
    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])
                    
    return x_train,y_train,x_test,y_test
    

def get_word_index(path):
    f = open(path)
    data = json.load(f)
    f.close()
    return data
    
    
    
if __name__ == '__main__':
    imfile = 'E:\pest1\imdb\imdb.npz'
    x_train,y_train,x_test,y_test = load_data(imfile)
    print(x_train)
    path = 'E:\pest1\imdb\imdb_word_index.json'
    data = get_word_index(path)
    print(data)