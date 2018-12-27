from keras import Input, Model

from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
from settings import *

from keras.models import Sequential
from keras.layers import LSTM, Reshape, concatenate, dot

from keras.layers import Dense, Activation, merge
from keras.layers import Embedding
from keras.callbacks import TensorBoard
import progressbar

import matplotlib

import data_processing
import numpy as np
import _pickle as pickle
import os
import data_class
import time

# settings

model_path = 'models/notes/skipGram'
model_filetype = '.pickle'

window_size = 3
vector_dim = 300
epochs = 1000000
train_set_size = 4
test_set_size = 1

valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

sampling_table = sequence.make_sampling_table(note_vocab_size)

train_set,test_set = data_class.get_note_train_and_test_set(train_set_size,test_set_size)
note_to_index, index_to_note = data_processing.get_note_dict()
# 预处理skip-gram 生成target,context 作为模型的输入

couples = []
labels = []
word_context = []
word_target = []
for i in range(len(train_set)):

    couple, label = skipgrams(train_set[i], note_vocab_size, window_size=window_size, sampling_table=sampling_table)
    target, context = zip(*couple)
    labels.append(label)
    couples.append(couple)
    word_target.append(np.array(target, dtype="int32"))
    word_context.append(np.array(context, dtype="int32"))

print(couples[:10], labels[:10])

# Models using keras functional API
input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(note_vocab_size, vector_dim, input_length=1, name='embedding')

# embedding lookup
target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

# cosine similarity
simi = dot([target, context], axes=0)

dot_product = dot([target, context], axes=1)
dot_product = Reshape((1,))(dot_product)

output = Dense(1, activation='sigmoid')(dot_product)

model = Model(input=[input_target, input_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

validation_model = Model(input=[input_target, input_context], output=simi)

class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = index_to_note[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % str(valid_word)
            for k in range(top_k):
                close_word = index_to_note[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((note_vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in range(note_vocab_size):
            in_arr1[0,] = valid_word_idx
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim


sim_cb = SimilarityCallback()


def save_embed_metadata(embedding_data,target_file):
    with open(target_file,'w') as f:
        for i in range(len(embedding_data)):
            try:
                f.write(str(embedding_data[i]) + "\n")
            except:
                print("wrong input")
    return

tbCallBack = TensorBoard(log_dir='/tmp/logs',  # log 目录
                         histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                         write_graph=True,  # 是否存储网络结构图
                         write_grads=True,  # 是否可视化梯度直方图
                         write_images=True)
tbCallBack.set_model(model)

def train():
    print('training model...')
    arr_1 = np.zeros((1,))
    arr_2 = np.zeros((1,))
    arr_3 = np.zeros((1,))
    for e in range(epochs):
        for  i in range(len(train_set)):
            idx = np.random.randint(0, len(labels[i]) - 1)
            arr_1[0,] = word_target[i][idx]
            arr_2[0,] = word_context[i][idx]
            arr_3[0,] = labels[i][idx]
            # loss = model.fit([arr_1, arr_2], arr_3,callbacks=[tbCallBack])
            loss = model.train_on_batch([arr_1, arr_2], arr_3)
        if e % 100 == 0:
            print("Iteration {}, loss={}".format(e, loss))
        if e % 10000 == 0:
            model_save_path = model_path + 'model_' + 'Epoch' + str(e) + model_filetype
            model.save(model_save_path)
            pickle.dump(model.layers[2].get_weights()[0],
                        open(model_path + 'embedding_' + 'Epoch' + str(e) + model_filetype, 'wb'))
            sim_cb.run_sim()

    tbCallBack.on_train_end()

print('start training')
train()
