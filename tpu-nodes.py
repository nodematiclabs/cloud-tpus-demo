import os

import tensorflow as tf

tpu = tf.distribute.cluster_resolver.TPUClusterResolver(os.environ["KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS"])
print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
