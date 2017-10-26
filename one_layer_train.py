
# coding: utf-8

# # One Layer Perceptron

# ## import tensorflow and mnist data

# In[1]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# ## training Data

# In[2]:

batch_xs, batch_ys = mnist.train.next_batch(5)


# In[3]:

batch_xs


# In[4]:

batch_xs.shape


# In[5]:

batch_xs[0]


# In[6]:

batch_ys


# In[7]:

batch_ys[0]


# ## testing data

# In[8]:

mnist.test.images


# In[9]:

mnist.test.images.shape


# In[10]:

mnist.test.labels


# In[11]:

mnist.test.labels.shape


# ## computational graph

# In[12]:

x_ = tf.placeholder(tf.float32, [None, 784], name="x_")
y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

W = tf.Variable(tf.zeros([784, 10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")
y = tf.nn.softmax(tf.matmul(x_, W) + b)

cross_entropy = -tf.reduce_mean(y_ * tf.log(y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
trainer = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()


# ## session

# In[13]:

sess = tf.Session()
sess.run(init)


# ## W

# In[14]:

sess.run(W)


# In[15]:

sess.run(W).shape


# ## b

# In[16]:

sess.run(b)


# In[17]:

sess.run(b).shape


# ## matmul(x_,w)

# In[18]:

result_matmul = sess.run(tf.matmul(x_, W),feed_dict={x_: batch_xs})


# In[19]:

result_matmul


# In[20]:

result_matmul.shape


# ## matmul(x_,w)+b

# In[21]:

result_matmul_add = sess.run(tf.matmul(x_, W)+b,feed_dict={x_: batch_xs})


# In[22]:

result_matmul_add


# In[23]:

result_matmul_add.shape


# ## y

# In[24]:

result_y = sess.run(y,feed_dict={x_: batch_xs})
result_y


# In[25]:

result_y.shape


# ## y_ * log(y)

# In[26]:

result_y_logy = sess.run(y_ * tf.log(y) ,feed_dict={x_: batch_xs, y_:batch_ys})
result_y_logy


# In[27]:

result_y_logy.shape


# ## cross entropy

# In[28]:

result_cross_entropy = sess.run(cross_entropy  ,feed_dict={x_: batch_xs, y_:batch_ys})
result_cross_entropy


# ## evaluation

# In[29]:

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ## correct prediction

# In[30]:

result_correct_prediction = sess.run(correct_prediction, feed_dict={x_: batch_xs, y_: batch_ys})
result_correct_prediction


# ## accurarcy

# In[31]:

result_accurarcy = sess.run(accuracy, feed_dict={x_: batch_xs, y_: batch_ys})
result_accurarcy


# ## training for 1 iteration

# In[32]:

sess.run(trainer,  feed_dict={x_: batch_xs, y_: batch_ys})


# ## cross entropy

# In[33]:

result_cross_entropy_2 = sess.run(cross_entropy  ,feed_dict={x_: batch_xs, y_:batch_ys})
result_cross_entropy_2


# ## accurarcy

# In[34]:

result_accurarcy_2 = sess.run(accuracy, feed_dict={x_: batch_xs, y_: batch_ys})
result_accurarcy_2


# ## training for 1000 iterations

# In[35]:

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(trainer,  feed_dict={x_: batch_xs, y_: batch_ys})
    if (i+1)%50 == 0:
        print "%s, %s" %(i+1, (sess.run(accuracy, feed_dict={x_: mnist.test.images, y_: mnist.test.labels})))


# ## save model

# In[36]:

saver = tf.train.Saver()


# In[37]:

saver.save(sess, "model.ckpt")


# In[ ]:

## getting value


# In[ ]:



