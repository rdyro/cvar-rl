def pred_op(in_, layerN, scope, out_nb):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    nn_ = in_
    # hidden layers
    for i in range(len(layerN)):
      nn_ = tf.dense(nn_, layerN[i], activation=tf.nn.relu)
    # output layer
    nn_ = tf.dense(nn_, out_nb, activation=tf.identity)
  return nn_
