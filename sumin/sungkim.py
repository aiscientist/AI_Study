
# coding: utf-8

# In[14]:

import tensorflow as tf
hello = tf.constant('Hello world')
print (hello)
sess = tf.Session()
print (sess.run(hello))


# In[6]:

#basic operation
import tensorflow as tf
sess = tf.Session()
a = tf.constant(3)
b = tf.constant(2)
c = a + b
print (c)
print (sess.run(c))


# In[12]:

#placeholder
import tensorflow as tf
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
c = tf.multiply(a, b)
sess = tf.Session()
print ('a * b = %i' %sess.run(c, feed_dict={a:3, b:2}))


# In[26]:

#linear regression
import tensorflow as tf
x_data = [1,2,3]
y_data = [1,2,3]
W = tf.Variable(tf.random_uniform([1],-1.0, 1.0))
b = tf.Variable(tf.random_uniform([1],-1.0, 1.0))

hypothesis = W * x_data +b
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
#변수 초기화 해야함.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#train
for step in range(2001):
    sess.run(train)
    if step % 200 == 0:
        print (step, sess.run(cost), sess.run(W), sess.run(b))


# In[25]:

#linear regression w/ placeholder--> 모델 재활용 가능
import tensorflow as tf
x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1],-1.0, 1.0))
b = tf.Variable(tf.random_uniform([1],-1.0, 1.0))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


hypothesis = W * X +b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 200 == 0:
        print (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b))
        
print (sess.run(hypothesis, feed_dict={X:5}))


# In[ ]:

#linear regression cost
import tensorflow as tf
#import matplotlib.pyplot as plt
X = [1.,2.,3.]
Y = [1.,2.,3.]
m = n_samples = len(X)

W = tf.placeholder(tf.float32)
hypothesis = tf.multiply(X, W)

cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2))/(m)

init = tf.global_variables_initializer()

w_val = []
cost_val = []

sess = tf.Session()
sess.run(init)
for i in range(-30, 50):
    print (i*0.1, sess.run(cost, feed_dict={W: i *0.1}))
    w_val.append(i*0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i*0.1}))
    
#graphic display
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()


# In[41]:

#multi-variable linear regression (입력 여러개)
x1_data = [1,0,3,0,5]
x2_data = [0,2,0,4,0]
y_data = [1,2,3,4,5]

w1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
w2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

#H(x)=W1X1 + W2X2 + b
hypothesis = w1 * x1_data + w2 * x2_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1) #learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

#변수 초기화 해야함.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#train
for step in range(2001):
    sess.run(train)
    if step % 200 == 0:
        print (step, sess.run(cost), sess.run(w1), sess.run(w2), sess.run(b))



# In[49]:

#matrix
x_data = [[1.,0.,3.,0.,5.],
          [0.,2.,0.,4.,0.]]
y_data = [1,2,3,4,5]

w = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
hypothesis = tf.matmul(w, x_data) + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1) #learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

#변수 초기화 해야함.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#train
for step in range(2001):
    sess.run(train)
    if step % 200 == 0:
        print (step, sess.run(cost), sess.run(w), sess.run(b))



# In[53]:

#matrix with b
x_data = [[1,1,1,1,1],
          [1.,2.,0.,4.,0.],
          [1.,0.,3.,0.,5.]]
y_data = [1,2,3,4,5]

w = tf.Variable(tf.random_uniform([1,3], -1.0, 1.0))
#b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
hypothesis = tf.matmul(w, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1) #learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

#변수 초기화 해야함.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#train
for step in range(2001):
    sess.run(train)
    if step % 200 == 0:
        print (step, sess.run(cost), sess.run(w))



# In[56]:

#load data from file
import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1];

print ('x', x_data)
print ('y', y_data)

w = tf.Variable(tf.random_uniform([1,3], -1.0, 1.0))
#b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
hypothesis = tf.matmul(w, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1) #learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

#변수 초기화 해야함.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#train
for step in range(2001):
    sess.run(train)
    if step % 200 == 0:
        print (step, sess.run(cost), sess.run(w))


# In[7]:

#linear regression cost
import tensorflow as tf
from matplotlib import pyplot as plt

X = [1.,2.,3.]
Y = [1.,2.,3.]
m = n_samples = len(X)

W = tf.placeholder(tf.float32)
hypothesis = tf.multiply(X, W)

cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2))/(m)

init = tf.global_variables_initializer()

W_val = []
cost_val = []

sess = tf.Session()
sess.run(init)
for i in range(-30, 50):
    print (i*0.1, sess.run(cost, feed_dict={W: i *0.1}))
    W_val.apped(sess.run(cost, feed_dict={W: i*0.1}nd(i*0.1)
    cost_val.appen))
    
#graphic display
plt.plot(W_val, cost_val, 'bo')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()


# In[13]:

#softmax
import tensorflow as tf
import numpy as np
xy = np.loadtxt('train2.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])
W = tf.Variable(tf.zeros([3,3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W))
learning_rate = 0.001

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    
    for step in range(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

            
    a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
    print (a, sess.run(tf.arg_max(a, 1)))


# In[17]:

#logistic classification 

from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf

xy = np.loadtxt('train3.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# x_data의 크기만큼 W 할당
W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))


h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))


learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})

    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
         
print ('========================================================')
        
print (sess.run(hypothesis, feed_dict={X:[[1], [2], [2]]}) > 0.5)
print (sess.run(hypothesis, feed_dict={X:[[1], [5], [5]]}) > 0.5)

print (sess.run(hypothesis, feed_dict={X:[[1,1], [4,3], [3,5]]}) > 0.5)


# In[ ]:



