import  os
import  tensorflow as tf
import  numpy as np
import csv
import seaborn
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  
plt.rcParams['axes.unicode_minus']=False  
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics,losses
tf.random.set_seed(22)

def load_data(root, mode='train'):
    attribute=[]
    label=[]
    file = open(root)
    reader = csv.reader(file)
    for row in reader:
        result1 = list(map(float, row[:-1]))
        attribute.append(result1)

        label.append(int(float(row[-1]))-1)
    
    return attribute,label

def preprocess(x,y):

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=3)
    return x, y

optimizer = optimizers.Adam(lr=1e-3)

batchsz = 20
attribute,label=load_data('D:\\Jupyter\\数值分析课设2\\wine.csv')
attribute,label=preprocess(attribute,label)

a=tf.linalg.inv(tf.matmul(tf.transpose(attribute), attribute))
b=tf.matmul(tf.transpose(attribute),label)
W=tf.matmul(a,b)
print(W)

attribute,label=load_data('D:\\Jupyter\\数值分析课设2\\wine.csv')
W1 = tf.Variable(W, name='Weights')
b1 = tf.Variable(tf.zeros([3]), name='Biases') 
prediction = tf.add(tf.matmul(attribute, W1), b1)
print(prediction.shape)
prediction = tf.nn.softmax(prediction, axis=1)
prediction = tf.argmax(prediction, axis=1)
prediction = tf.cast(prediction, dtype=tf.int32)

correct = tf.equal(prediction, label)
correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

acc = correct / len(attribute)
print(acc)
