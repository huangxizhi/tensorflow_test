#coding=utf-8
import tensorflow as tf
import numpy as np

#数据准备
data_len = 100
rng = np.random.RandomState(112233)
X = rng.rand(data_len, 2)
Y = [[int(x + y < 1)] for x, y in X]

# 前向传播
x = tf.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random_normal(shape=(2, 3), mean=0, stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal(shape=(3, 1), mean=0, stddev=1, seed=1))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
loss = tf.reduce_mean(tf.square(y - y_))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 迭代，反向传播
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    epoch = 3000
    for i in range(epoch):
        batch_size = 32
        start, end = 0, batch_size

        while end <= data_len:
            sess.run(train_op, feed_dict={x : X[start: end], y_ : Y[start: end]})
            start, end = end + 1, end + batch_size

        if i % 300 == 0:
            print('loss', sess.run(loss, feed_dict={x : X, y_ : Y}))    # 一次性将所有训练数据喂入计算loss

    print(sess.run(w1), sess.run(w2))


