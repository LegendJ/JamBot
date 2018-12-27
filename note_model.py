from tensorflow.contrib.tensorboard.plugins import projector

from settings import *
from keras.models import load_model
import keras
import os
import numpy as np
from numpy import array
import tensorflow as tf
import _pickle as pickle
from keras import backend as K

from data_processing import get_note_dict


class Note_Model:

    def __init__(self,
                 model_path,
                 prediction_mode='sampling',
                 first_notes=[1, 3, 2, 1, 1, 3, 2, 1],
                 resample='none',
                 dim_factor=2,
                 temperature=1.0):

        print('loading note model ...')

        self.model = keras.models.load_model(model_path)
        self.model.reset_states()
        self.embed_layer_output = K.function([self.model.layers[0].input], [self.model.layers[0].output])
        self.embed_model = keras.models.Model(inputs=self.model.input,
                                              outputs=self.model.get_layer(name="embedding").output)
        self.note_to_index, self.index_to_notes = get_note_dict()
        self.prediction_mode = prediction_mode
        self.temperature = temperature
        self.resample = resample
        self.dim_factor = dim_factor
        self.song = []

        for note in first_notes[:-1]:
            #            print(note)
            self.model.predict(array([[note]]))
            self.song.append(note)

        note = first_notes[-1]

        self.song.append(note)
        self.current_note = array([[note]])

    def predict_next(self):

        prediction = self.model.predict(self.current_note)[0]

        if self.resample == 'hard':
            prediction[self.current_note] = 0
            prediction = prediction / np.sum(prediction)

        elif self.resample == 'soft':
            prediction[self.current_note] /= self.dim_factor
            prediction = prediction / np.sum(prediction)

        #        print(prediction)
        prediction = np.log(prediction) / self.temperature
        prediction = np.exp(prediction) / np.sum(np.exp(prediction))

        if self.prediction_mode == 'argmax':
            #            print('argmax')
            while True:
                next_note = np.argmax(prediction)
                if next_note != 0:
                    break
        #            print(next_note)

        elif self.prediction_mode == 'sampling':
            while True:
                next_note = np.random.choice(len(prediction), p=prediction)
                #                print(next_note)
                if next_note != 0:
                    break

        #            print(next_note)

        self.song.append(next_note)
        self.current_note = np.array([next_note])
        return self.current_note[0]

    def embed_note(self, note):

        return self.embed_layer_output([[[note]]])[0][0][0]

    def embed_notes_song(self, notes):

        embeded_notes = []

        for note in notes:
            embeded_notes.append(self.embed_note(note))

        return embeded_notes


class Embed_Note_Lstm_Model:

    def __init__(self, model_path):
        print('loading note model ...')

        model = keras.models.load_model(model_path)
        model.reset_states()
        self.weights = model.layers[0].get_weights()[0]
        self.embed_layer_output = K.function([model.layers[0].input], [model.layers[0].output])
        self.note_to_index, self.index_to_notes = get_note_dict()

    def embed_note(self, note):
        return self.embed_layer_output([[[note]]])[0][0][0]

    def embed_notes_song(self, notes):
        embeded_notes = []

        for note in notes:
            embeded_notes.append(self.embed_note(note))

        return embeded_notes

class Embed_Note_SG_Model:

    def __init__(self,model_path):
        print('loading note model ...')

        model = keras.models.load_model(model_path)
        model.reset_states()
        self.weights = model.layers[0].get_weights()[0]
        self.embed_layer_output = K.function([model.layers[0].input], [model.layers[0].output])
        self.note_to_index, self.index_to_notes = get_note_dict()

    def embed_note(self, note):
        return self.embed_layer_output([[[note]]])[0][0][0]

    def embed_notes_song(self, notes):
        embeded_notes = []

        for note in notes:
            embeded_notes.append(self.embed_note(note))

        return embeded_notes

if __name__ == "__main__":
    # Paths:
    model_folder = '/home/just/PycharmProjects/JamBot/models/notes/1545844070-Shifted_True_Lr_1e-05_EmDim_88_opt_Adam_bi_False_lstmsize_512_trainsize_4_testsize_1_samples_per_bar8/model_Epoch50_3.pickle'
    model_name = 'modelEpoch10'

    model = Embed_Note_Lstm_Model(model_folder)

    LOG_DIR='/tmp/logs'

    arr_embeds = model.weights

    embedding_ph = tf.placeholder(tf.float32, arr_embeds.shape, name='embedding_place_holder')
    embedding_var = tf.Variable(tf.zeros(arr_embeds.shape), name='note_embedding')
    var_init = embedding_var.assign(embedding_ph)
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    #  Create projector
    config = projector.ProjectorConfig()

    #  Add embedding to the projector
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # embedding.metadata_path = path_for_metadata
    projector.visualize_embeddings(summary_writer, config)

    print("Saving Data")
    # Save the data
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(var_init, feed_dict={embedding_ph: arr_embeds})
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, 'note_embedding' + "_model.ckpt"), 1)

    # for i in range(0, 16):
    #     model.predict_next()
    # print(model.song)
