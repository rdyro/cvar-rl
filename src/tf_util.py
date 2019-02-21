from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

def pred_op(in_, layerN, scope, out_nb, keep_prob=None):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    nn_ = in_
    # hidden layers
    for i in range(len(layerN)):
      nn_ = tf.layers.dense(nn_, layerN[i], activation=tf.nn.tanh)
      nn_ = nn_ if keep_prob == None else tf.nn.dropout(nn_, keep_prob)
    # output layer
    nn_ = tf.layers.dense(nn_, out_nb, activation=tf.identity)
  return nn_

def assign_op(scope_to, scope_from):
  vars_to = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_to)
  vars_from = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
      scope=scope_from)
  assert len(vars_to) == len(vars_from)
  ops = [tf.assign(vars_to[i], vars_from[i]) for i in range(len(vars_to))]
  return tf.group(*ops)

def loss_op(v1_, v2_):
  return tf.reduce_mean(tf.squared_difference(v1_, v2_))

def optimizer_op(loss_, scope, lr, clip_gradient=None):
  adam = tf.train.AdamOptimizer(learning_rate=lr)
  s_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
  gs_vs = adam.compute_gradients(loss_, var_list=s_vars)
  gs = [el[0] for el in gs_vs]
  vs = [el[1] for el in gs_vs]
  if clip_gradient != None:
    gs = [tf.clip_by_norm(el, clip_gradient) for el in gs]
    gs_vs = list(zip(gs, vs))
  train_op = adam.apply_gradients(gs_vs)
  grad_norm = tf.global_norm(gs)
  return (train_op, grad_norm)
