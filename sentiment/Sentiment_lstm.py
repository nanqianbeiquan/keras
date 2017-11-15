# -*- coding: utf-8 -*-

import yaml
import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
from sklearn.cross_validation import train_test_split
import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
import jieba
import pandas as pd
#import sys
sys.setrecursionlimit(1000000)

vocab_dim = 100
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 10
window_size = 7
batch_size = 32
n_epoch = 4
input_length = 100
cpu_count = multiprocessing.cpu_count()


# 加载训练文件
def loadfile():
    neg = pd.read_excel('data/neg.xls',header = None, index = None)
    pos = pd.read_excel('data/pos.xls',header = None, index = None)
    
    combined = np.concatenate((pos[0],neg[0]))
    y = np.concatenate((np.ones(len(pos),dtype=int),np.zeros(len(neg),dtype=int)))
    return combined,y

# 对句子进行分词，并去掉换行符
def tokenizer(text):
    text = [jieba.lcut(document.replace('\n','')) for document in text]
    return text

#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model = None,combined = None):
    if (combined is not None) and (model is not None):
        genism_dict = Dictionary()
        genism_dict.doc2bow(model.vocab.keys(),allow_update = True)
        w2indx = {v:k+1 for k,v in genism_dict.items()}      # 所有频数超过10的索引
        w2vec = {word:model[word] for word in w2indx.keys()} # 所有频数超过10的词向量
        
        def parse_dataset(combined):
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined = parse_dataset(combined)
        combined = sequence.pad_sequence(combined,maxlen = maxlen) #每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx,w2vec,combined
    else:
        print('No data provided...')
        
#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
    print(vocab_dim,n_exposures,window_size,cpu_count,n_iterations)
    model = Word2Vec(size = vocab_dim,
                     min_count = n_exposures,
                     window = window_size,
                     workers = cpu_count,
                     iter = n_iterations)
    model.build_vocab(combined)
    model.train(combined,total_examples=100,epochs=10)
    model.save('lstm_data/Word2vec_model.h5')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return index_dict, word_vectors,combined

def get_data(index_dict,word_vectors,combined,y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    print (x_train.shape,y_train.shape)
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test  

##定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print ('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print ('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print ("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch,verbose=1, validation_data=(x_test, y_test),show_accuracy=True)

    print ("Evaluate...")
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('lstm_data/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('lstm_data/lstm.h5')
    print ('Test score:', score)


#训练模型，并保存
def train():
    print ('Loading Data...')
    combined,y=loadfile()
    print (len(combined),len(y))
    print ('Tokenising...')
    combined = tokenizer(combined)
    print ('Training a Word2vec model...')
    index_dict, word_vectors,combined=word2vec_train(combined)
    print ('Setting up Arrays for Keras Embedding Layer...')
    n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
    print (x_train.shape,y_train.shape)
    train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)


def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('lstm_data/Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined  

def lstm_predict(string):
    print ('loading model......')
    with open('lstm_data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    print(yaml_string)
    model = model_from_yaml(yaml_string)
    
    print ('loading weights......')
    
if __name__ == '__main__':
    train()


