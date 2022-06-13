import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

#%%
data_o = np.empty(shape=(1,96))
load_data = open(r'../data_after_process_weekday.pickle','rb')
data_weekday = pickle.load(load_data)
for key in data_weekday:
    _ = data_weekday[key]
    data_o = np.vstack((data_o,_[:,1].reshape(1,96)))
    
data_o = data_o[1:,:]
# for i in range(len(data_o)):
#     plt.plot(data_o[:,i])
#     plt.show()
             
# def y(x):
#     z = x**2
#     return z
# xx = np.random.uniform(-10., 10., size=[500, 96])
# xx.sort()
# data_o = y(xx)

#%% 
'Network parameters d'
n_input_g = 96
h1_g = 300
h2_g = 300
n_output_g = 96

x_g = tf.placeholder("float",[None, n_input_g])
y_g = tf.placeholder("float",[None, n_output_g])

def Gen(x_g,weights_g,biases_g):
    layer_1 = tf.add(tf.matmul(x_g, weights_g['h1_g']), biases_g['b1_g'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1,weights_g['h2_g']), biases_g['b2_g'])
    layer_2 = tf.nn.relu(layer_2)
    out_log = tf.matmul(layer_2,weights_g['out_g'] + biases_g['out_g'])
    # Gen_pro = tf.nn.sigmoid(out_log)
    return out_log

weights_g ={
    'h1_g':tf.Variable(tf.random_normal([n_input_g, h1_g])),
    'h2_g':tf.Variable(tf.random_normal([h1_g, h2_g])),
    'out_g':tf.Variable(tf.random_normal([h2_g, n_output_g]))
}

biases_g = {
    'b1_g': tf.Variable(tf.random_normal([h1_g])),
    'b2_g': tf.Variable(tf.random_normal([h2_g])),
    'out_g': tf.Variable(tf.random_normal([n_output_g]))
}

'Network parameters g'
n_input_d = 96
h1_d =96
h2_d =96
n_output_d = 1

x_d = tf.placeholder("float",[None, n_input_d])
y_d = tf.placeholder("float",[None, n_output_d])

def Ds(x_d,weights_d,biases_d):
    layer_1 = tf.add(tf.matmul(x_d, weights_d['h1_d']), biases_d['b1_d'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1,weights_d['h2_d']), biases_d['b2_d'])
    layer_2 = tf.nn.relu(layer_2)
    out_logit = tf.matmul(layer_2,weights_d['out_d'] + biases_d['out_d'])
    D_prob = tf.nn.sigmoid(out_logit)
    # D_prob = tf.nn.tanh(out_logit)
    return out_logit,D_prob

weights_d ={
    'h1_d':tf.Variable(tf.random_normal([n_input_d, h1_d])),
    'h2_d':tf.Variable(tf.random_normal([h1_d, h2_d])),
    'out_d':tf.Variable(tf.random_normal([h2_d, n_output_d]))
}

biases_d = {
    'b1_d': tf.Variable(tf.random_normal([h1_d])),
    'b2_d': tf.Variable(tf.random_normal([h2_d])),
    'out_d': tf.Variable(tf.random_normal([n_output_d]))
}

def sample_gen(m, n): #生成维度为[m, n]的随机噪声作为生成器G的输入
    return np.random.uniform(-1., 1., size=[m, n])



#%%
learning_rate = 0.00001  #学习率
mb_size = 10

#%%
G_sample = Gen(x_g,weights_g,biases_g)  #取得生成器的生成结果
D_real, D_logit_real = Ds(x_d,weights_d,biases_d)  #取得判别器判别的真实手写数字的结果
D_fake, D_logit_fake = Ds(G_sample,weights_d,biases_d) #取得判别器判别的生成的手写数字的结果

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_logit_real))) #对判别器对真实样本的判别结果计算误差(将结果与1比较)
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_logit_fake))) #对判别器对虚假样本(即生成器生成的手写数字)的判别结果计算误差(将结果与0比较)
D_loss = (D_loss_real + D_loss_fake) #判别器的误差
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_logit_fake))) #生成器的误差(将判别器返回的对虚假样本的判别结果与1比较)
# D_loss = tf.reduce_mean(tf.cast(-tf.log(D_real)-tf.log(1-D_fake),'float'))
# G_loss = tf.reduce_mean(tf.cast(-tf.log(D_fake),'float'))

D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss)
G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

i = 0
loss_d = []
loss_g = []

with tf.Session() as  sess:
    sess.run(init)
    for it in range(1500000): #训练100万次
        if it % 1000 == 0: #每训练1000次就保存一下结果
            data = sample_gen(1,96)
            # print(data)
            samples = sess.run(G_sample, feed_dict={x_g:data})
            print(samples)
            i += 1          
            plt.plot(samples.reshape(96,1))
            plt.show()
            save_path = saver.save(sess, "folder_for_nn/save_net.ckpt")

        sampe = sample_gen(10,96)
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={x_d: data_o, x_g: sampe})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={x_d: data_o, x_g: sampe})
        loss_d.append(D_loss_curr)
        loss_g.append(G_loss_curr)
        if it % 100 == 0:
            print(it,D_loss_curr,G_loss_curr)

# with tf.Session() as  sess:
#     sess.run(init)
#     for it in range(150000): #训练100万次
#         if it % 1000 == 0: #每训练1000次就保存一下结果
#             data = np.random.uniform(-10., 10., size=[1, 96])
#             # data = y(np.random.uniform(-10., 10., size=[1, 96]))
#             # print(data)
#             samples = sess.run(G_sample, feed_dict={x_g:data})
#             # print(samples)
#             plt.plot(samples.reshape(96,1))
#             plt.show()
            
#         sampe = np.random.uniform(-10., 10., size=[100, 96])
#         data_o = y(np.random.uniform(-10., 10., size=[100, 96]))
#         _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={x_d: data_o, x_g: sampe})
#         _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={x_d: data_o, x_g: sampe})
#         loss_d.append(D_loss_curr)
#         loss_g.append(G_loss_curr)
#         print(it,D_loss_curr,G_loss_curr)

plt.plot(loss_d[-100000:])
plt.plot(loss_g[-100000:])
plt.show()

        


