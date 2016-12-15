import os.path
import tensorflow as tf

from tensorflow.contrib.session_bundle import exporter
import utils
import vgg

def export():
  with tf.Graph().as_default():
    # Build inference model.
    # Please refer to Tensorflow inception model for details.

    # Input transformation.
    jpegs = tf.placeholder(tf.string)
    images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)
    print(images)
    # Run inference.
    feature = vgg.inference(images)

    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess, 'model/inshop.sgd.adam')
      # Export inference model.
      init_op = tf.group(tf.initialize_all_tables(), name='init_op')

      model_exporter = exporter.Exporter(saver)
      signature = exporter.classification_signature(
          input_tensor=jpegs, classes_tensor=None, scores_tensor=feature)
      model_exporter.init(default_graph_signature=signature, init_op=init_op)
      model_exporter.export('model', tf.constant(150000), sess)
      print('Successfully exported model to model/.')


def preprocess_image(image_buffer):
  # decode and resize
  image = tf.image.decode_jpeg(image_buffer, channels=3)
  image = utils.preprocess_image(image)
  #image = tf.expand_dims(image, 0)
  return image


export()
