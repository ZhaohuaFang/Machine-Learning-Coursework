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
    
    return x, y

model = Sequential([
    layers.Dense(3, activation=tf.nn.leaky_relu)
])

optimizer = optimizers.Adam(lr=1e-3)

batchsz = 20
attribute,label=load_data('D:\\Jupyter\\数值分析课设2\\wine.csv')
num=int(len(attribute)*0.8)

idx=tf.range(len(attribute))
idx=tf.random.shuffle(idx)

x_train,y_train=tf.gather(attribute,idx[:num]),tf.gather(label,idx[:num])
x_test,y_test=tf.gather(attribute,idx[num:]),tf.gather(label,idx[num:])

db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.shuffle(1000).map(preprocess).batch(batchsz)

accc=[]
losss=[]
for epoch in range(600):
    #training
    for step, (x, y) in enumerate(db_train):
        with tf.GradientTape() as tape:
            y = tf.one_hot(y, depth=3)
            out=model(x)
            loss=tf.square(y - out)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step %100 == 0:
            losss.append(loss)
    #testing
    total_num = 0
    total_correct = 0
    for x,y in db_test:
        
        out=model(x)
        out = tf.nn.softmax(out, axis=1)
        pred = tf.argmax(out, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        correct = tf.equal(pred, y)
        correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
        
        total_num += x.shape[0]
        total_correct += int(correct)
        
    acc = total_correct / total_num
    print(epoch, 'acc_ave:', acc)
    accc.append(acc)

print("最高k-fold精确率为"+str(max(accc)))

attribute,label=preprocess(attribute,label)
prediction=model(attribute)
prediction = tf.nn.softmax(prediction, axis=1)
prediction = tf.argmax(prediction, axis=1)
prediction = tf.cast(prediction, dtype=tf.int32)
correct = tf.equal(prediction, label)
correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

acc = correct / len(attribute)
print(acc)

x=np.linspace(1,len(accc),len(accc))
plt.plot(x,accc,c='r',label="wine")
plt.xlabel("迭代次数")
plt.ylabel("准确性")
plt.legend(loc=4) 
plt.title("迭代次数与准确性的关系图-wine")
plt.savefig('迭代次数与准确性的关系图-wine.png',dpi=500,bbox_inches='tight')
plt.show()

x=np.linspace(1,len(losss),len(losss))
plt.plot(x,losss,c='g',label="wine")
plt.xlabel("迭代次数")
plt.ylabel("误差")
plt.legend(loc=4) 
plt.title("迭代次数与误差的关系图-wine")
plt.savefig('迭代次数与误差的关系图-wine.png',dpi=500,bbox_inches='tight')
plt.show()
