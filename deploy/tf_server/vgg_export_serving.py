import os.path
import tensorflow as tf

from tensorflow.contrib.session_bundle import exporter
import utils
import vgg

def export():
  with tf.Graph().as_default():
    #TODO(xuesen) for serving
    serialized_tf_example = tf.placeholder(tf.string,name='tf_example')
    feature_configs={'image/encoded':tf.FixedLenFeature(shape=[],dtype=tf.string),}
    tf_example = tf.parse_example(serialized_tf_example,feature_configs)

    jpegs = tf_example['image/encoded']
    images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)
    # Run inference.
    feature = vgg.inference(images)

    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess, 'model/inshop.sgd.adam')
      # Export inference model.
      init_op = tf.group(tf.initialize_all_tables(), name='init_op')


      #TODO() Export inference model using regression_signture ?
      feat_signature = exporter.regression_signature(
          input_tensor=serialized_tf_example,
          output_tensor=feature)
      named_graph_signature = {
              'inputs':exporter.generic_signature({'images':jpegs}),
              'outputs':exporter.generic_signature({'feats':feature})
              }
      model_exporter = exporter.Exporter(saver)
      model_exporter.init(default_graph_signature=feat_signature,
                          init_op=init_op,
                          named_graph_signatures=named_graph_signature)
      model_exporter.export('model/vgg_serving', tf.constant(150000), sess)
      print('Successfully exported model to model/.')


def preprocess_image(image_buffer):
  # decode and resize
  image = tf.image.decode_jpeg(image_buffer, channels=3)
  image = utils.preprocess_image(image)
  #image = tf.expand_dims(image, 0)
  return image


export()
