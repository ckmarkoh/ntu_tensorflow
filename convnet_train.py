
# coding: utf-8

# # Convolutional Neural Networks

# ## import tensorflow and mnist data

# In[1]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# ## training data

# In[2]:

batch_xs, batch_ys = mnist.train.next_batch(5)


# ## computational graph

# In[3]:

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# In[4]:

x_ = tf.placeholder(tf.float32, [None, 784], name="x_")
y_ = tf.placeholder(tf.float32, [None, 10], name="y_")


x_image = tf.reshape(x_, [-1,28,28,1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y= tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

optimizer = tf.train.AdamOptimizer(1e-4)
trainer = optimizer.minimize(cross_entropy)

init = tf.initialize_all_variables()


# ## session

# In[5]:

sess = tf.Session()
sess.run(init)


# ## conv1

# In[6]:

result_h_conv1 = sess.run(h_conv1,  feed_dict={x_: batch_xs})
result_h_conv1.shape


# ## pool1

# In[7]:

result_h_pool1 = sess.run(h_pool1,  feed_dict={x_: batch_xs})
result_h_pool1.shape


# ## conv2

# In[8]:

result_h_conv2 = sess.run(h_conv2,  feed_dict={x_: batch_xs})
result_h_conv2.shape


# ## pool2

# In[9]:

result_h_pool2 = sess.run(h_pool2,  feed_dict={x_: batch_xs})
result_h_pool2.shape


# ## fc1

# In[10]:

result_h_fc1 = sess.run(h_fc1,  feed_dict={x_: batch_xs})
result_h_fc1.shape


# ## y

# In[11]:

result_y = sess.run(y,  feed_dict={x_: batch_xs, keep_prob:0.5})
result_y.shape


# ## evaluation

# In[12]:

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ## saver

# In[13]:

saver = tf.train.Saver()


# ## training

# In[14]:

patience = 20
best_accurarcy = 0
i = 0
batch_size = 100

while True:
    i += 1
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(trainer,  feed_dict={x_: batch_xs, y_: batch_ys, keep_prob:0.5})
    if i%100 == 0:
        valid_accurarcy = 0
        for _ in range(mnist.validation.num_examples/batch_size):
            batch_xs, batch_ys = mnist.validation.next_batch(batch_size)
            valid_accurarcy += sess.run(accuracy, 
                feed_dict={x_: batch_xs, y_: batch_ys, keep_prob:1})
        valid_accurarcy = valid_accurarcy/(mnist.validation.num_examples/batch_size)
        print "%s, valid_accurarcy:%s" %(i, valid_accurarcy)
        if valid_accurarcy > best_accurarcy:
            patience = 20
            best_accurarcy = valid_accurarcy
            print "save model"
            saver.save(sess, "model_conv.ckpt")
        else:
            patience -= 1
            if patience == 0:
                print "early stop"
                break


# In[16]:

valid_accurarcy = 0
test_accurarcy = 0
for _ in range(mnist.validation.num_examples/batch_size):
    batch_xs, batch_ys = mnist.validation.next_batch(batch_size)
    valid_accurarcy += sess.run(accuracy, feed_dict={x_: batch_xs, y_: batch_ys, keep_prob:1})
valid_accurarcy = valid_accurarcy/(mnist.validation.num_examples/batch_size)
for _ in range(mnist.test.num_examples/batch_size):
    batch_xs, batch_ys = mnist.test.next_batch(batch_size)
    test_accurarcy += sess.run(accuracy, feed_dict={x_: batch_xs, y_: batch_ys, keep_prob:1})
test_accurarcy = test_accurarcy/(mnist.test.num_examples/batch_size)

print "valid:%s, test:%s"%(valid_accurarcy,test_accurarcy)


# In[ ]:




# In[ ]:



