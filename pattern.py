from scipy import misc
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import glob
from PIL import Image
from utils import show_graph
import sys
import os
# preparing data
ellipse_fs = glob.glob('/home/abdullah/PycharmProjects/Pattern/Ellipse_TS/*.bmp')
line_fs = glob.glob('/home/abdullah/PycharmProjects/Pattern/Ellipse_TS/*.bmp')
Diamond_fs = glob.glob('/home/abdullah/PycharmProjects/Pattern/Ellipse_TS/*.bmp')

x = np.array([np.array(Image.open(fname)) for fname in ellipse_fs])
x = np.concatenate((x,np.array([np.array(Image.open(fname)) 
                                for fname in line_fs])),axis=0)
x = np.concatenate((x,[np.array(Image.open(fname)) 
                    for fname in Diamond_fs]),axis=0).astype('float32')

y=np.zeros((255,)).astype('int')
y[0:85]=0
y[85:170]=1
y[170:]=2
b = np.zeros((255, 3))
b[np.arange(255), y] = 1
y=b
X_train, X_test, y_train, y_test = train_test_split\
    (x, y, test_size=0.1)
X_train, X_eval, y_train, y_eval = train_test_split\
    (X_train, y_train, test_size=0.12)

# building model
def dense_relu_drop(x, phase,size, scope):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, size, 
                                               activation_fn=tf.nn.relu,
                                               scope='dense')
        dropout_2 =tf.layers.dropout(inputs=h1, rate=0.4,
                                     training=phase)

        
        return dropout_2

x = tf.placeholder('float32', (None, 256,256), name='x')
y = tf.placeholder('float32', (None,3), name='y')
phase = tf.placeholder(tf.bool, name='phase')
input_layer = tf.reshape(x, [-1, 256* 256])
# dense = tf.layers.dense(inputs=input_layer, units=32,activation=tf.nn.relu)
# bn1= tf.layers.batch_normalization(dense, 
#                                       center=True, trainable=True,scale=True)
# dropout_1 =tf.layers.dropout(inputs=bn1, rate=0.4,training=mode == tf.estimator.ModeKeys.TRAIN)
# dense2 = tf.layers.dense(inputs=dropout_1, units=64,activation=tf.nn.relu)
# dropout_2 =tf.layers.dropout(inputs=dense2, rate=0.4,training=mode == tf.estimator.ModeKeys.TRAIN)
# logits = tf.layers.dense(inputs=dropout_2, units=3)
h1 = dense_relu_drop(input_layer, phase,32,'layer1')
h2 = dense_relu_drop(h1, phase,64, 'layer2')
h3 = dense_relu_drop(h1, phase,64, 'layer3')
logits = tf.contrib.layers.fully_connected(h3, 3, 
                                             activation_fn=None,
                                             scope=scope)

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)), 
            'float32'))

with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))






# training 
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

history = []
iterep = 500
for i in range(iterep * 30):
    idx = np.random.choice(np.arange(len(y_train)), 100, replace=False)
    x_batch = X_train[idx]
    y_batch = y_train[idx]
    
    sess.run(train_step,
             feed_dict={'x:0': x_batch, 
                        'y:0': y_batch, 
                        'phase:0': 1})
    if (i + 1) %  iterep == 0:
        epoch = (i + 1)/iterep
        tr = sess.run([loss, accuracy], 
                      feed_dict={'x:0': X_train,
                                 'y:0': y_train,
                                 'phase:0': 1})
        t = sess.run([loss, accuracy], 
                     feed_dict={'x:0': X_eval,
                                'y:0': y_eval,
                                'phase:0': 0})
        history += [[epoch] + tr + t]
        print history[-1]

sys.path.append('/home/abdullah/PycharmProjects/Pattern') # point to your tensorflow dir
show_graph(tf.get_default_graph().as_graph_def())
