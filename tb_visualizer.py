#!/usr/bin/python
import sys
import tensorflow as tf
import numpy as np
import os
import pickle
import note_model
from note_model import *
import data_processing
from tensorflow.contrib.tensorboard.plugins import projector


name = sys.argv[1]
model_path = sys.argv[2]
print('get {} model under {}'.format(name,model_path))
LOG_DIR = os.path.join(os.getcwd(), "/tmp/logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

NAME_TO_VISUALISE_VARIABLE = name

path_for_metadata = os.path.join(LOG_DIR, name + '_metadata.tsv')
dict_path = 'data/note_dict_shifted.pickle'


with open(path_for_metadata, 'w') as f:
    note_2_index,index_2_note = data_processing.get_note_dict()
    for k,v in note_2_index.items():
        f.write(f"{repr(v)+' ',repr(k)}\n")
print("Metadata file created")


def set_model(var,model_path):
    return {
        'sg': lambda model_path :Embed_Note_SG_Model(model_path),
        'lstm' : lambda model_path :Embed_Note_Lstm_Model(model_path)
    }[var](model_path)


model = set_model(var = name,model_path= model_path)


if model == 'error':
    print('wrong input for name')
    sys.exit(0)
arr_embeds = model.weights

#  Set to a tf variable
embedding_ph = tf.placeholder(tf.float32, arr_embeds.shape, name='embedding_place_holder')
embedding_var = tf.Variable(tf.zeros(arr_embeds.shape), name=NAME_TO_VISUALISE_VARIABLE)
var_init = embedding_var.assign(embedding_ph)
summary_writer = tf.summary.FileWriter(LOG_DIR)

#  Create projector
config = projector.ProjectorConfig()

#  Add embedding to the projector
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = path_for_metadata
projector.visualize_embeddings(summary_writer, config)

print("Saving Data")
# Save the data
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(var_init, feed_dict={embedding_ph: arr_embeds})
saver = tf.train.Saver()
saver.save(sess, os.path.join(LOG_DIR, name + "_model.ckpt"), 1)

