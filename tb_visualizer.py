import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.tensorboard.plugins import projector


name = "tensorboard_test"
LOG_DIR = os.path.join(os.getcwd(), "/tmp/logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

NAME_TO_VISUALISE_VARIABLE = name

n_samples = 4824
n_features = 13

path_for_metadata = os.path.join(LOG_DIR, name + '_metadata.tsv')
with open(path_for_metadata, 'w') as f:
    for i in range(n_samples):
        f.write(f"{repr(i)}\n")
print("Metadata file created")
arr_embeds = np.random.rand(n_samples, n_features)

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

