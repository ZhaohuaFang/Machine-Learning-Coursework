batchsz = 20
alpha=0.8
dep=3
filename='C:\\Users\\Asus\\Desktop\\机器学习\\方向一数据\\wine.csv'
label="wine"
title="改进-迭代次数与准确性的关系图-wine"
save_png='改进-迭代次数与准确性的关系图-wine.png'

import seaborn
color=seaborn.xkcd_rgb['red']

import  os
import  tensorflow as tf
import  numpy as np
import csv
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
    
    return x, y

def network(dep):
    model=Sequential([
    layers.Dense(10, activation=tf.nn.leaky_relu), layers.Dropout(0.2), 
    layers.Dense(10, activation=tf.nn.leaky_relu), 
    layers.Dense(dep)])

    return model

model1 = network(dep)
model2 = network(dep)
model3 = network(dep)
model4 = network(dep)
model5 = network(dep)

optimizer1 = optimizers.Adam(lr=1e-4)
optimizer2 = optimizers.Adam(lr=1e-4)
optimizer3 = optimizers.Adam(lr=1e-4)
optimizer4 = optimizers.Adam(lr=1e-4)
optimizer5 = optimizers.Adam(lr=1e-4)

def create_k_fold_db(filename,batchsz,alpha):
    attribute,label=load_data(filename)
    num=int(len(attribute)*alpha)
    #k-fold cross validation
    idx=tf.range(len(attribute))
    idx=tf.random.shuffle(idx)
    
    x_train,y_train=tf.gather(attribute,idx[:num]),tf.gather(label,idx[:num])
    x_test,y_test=tf.gather(attribute,idx[num:]),tf.gather(label,idx[num:])

    db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)

    db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    db_test = db_test.shuffle(1000).map(preprocess).batch(batchsz)

    return db_train,db_test

db_train1,db_test1=create_k_fold_db(filename,batchsz,alpha)
db_train2,db_test2=create_k_fold_db(filename,batchsz,alpha)
db_train3,db_test3=create_k_fold_db(filename,batchsz,alpha)
db_train4,db_test4=create_k_fold_db(filename,batchsz,alpha)
db_train5,db_test5=create_k_fold_db(filename,batchsz,alpha)

def train(db_train,model,optimizer,dep):
    for step, (x, y) in enumerate(db_train):
        with tf.GradientTape() as tape:
            y = tf.one_hot(y, depth=dep)
            out=model(x)
            loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=out)
            loss = tf.reduce_mean(loss)

            #l2 regularization
            t=[]
            for i in range(len(model.trainable_variables)):        
                t.append(tf.nn.l2_loss(model.trainable_variables[i]))
            t=tf.reduce_sum(tf.stack(t))

            loss=loss+0.001*t
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

def test(db_test,model):
    total_num = 0
    total_correct = 0
    for x,y in db_test:
        out=model(x)
        out=tf.nn.sigmoid(out)
        out = tf.nn.softmax(out, axis=1)
        pred = tf.argmax(out, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        correct = tf.equal(pred, y)
        correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
        
        total_num += x.shape[0]
        total_correct += int(correct)  
    acc = total_correct / total_num
    return acc

accc=[]
for epoch in range(3000):
    acc_ave=0
    #training
    train(db_train1,model1,optimizer1,dep)
    #testing
    acc=test(db_test1,model1)
    acc_ave+=acc

    #training
    train(db_train2,model2,optimizer2,dep)
    #testing
    acc=test(db_test2,model2)
    acc_ave+=acc

    #training
    train(db_train3,model3,optimizer3,dep)
    #testing
    acc=test(db_test3,model3)
    acc_ave+=acc

    #training
    train(db_train4,model4,optimizer4,dep)
    #testing
    acc=test(db_test4,model4)
    acc_ave+=acc

    #training
    train(db_train5,model5,optimizer5,dep)
    #testing
    acc=test(db_test5,model5)
    acc_ave+=acc

    acc_ave=acc_ave/5
    accc.append(acc_ave)
    print(epoch, 'acc_ave:', acc_ave)

print("最高k-fold精确率为"+str(max(accc)))

def acc_total(model):
    attribute,label=load_data(filename)
    attribute,label=preprocess(attribute,label)
    prediction=model(attribute)
    prediction=tf.nn.sigmoid(prediction)
    prediction = tf.nn.softmax(prediction, axis=1)
    prediction = tf.argmax(prediction, axis=1)
    prediction = tf.cast(prediction, dtype=tf.int32)
    correct = tf.equal(prediction, label)
    correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
    acc = correct / len(attribute)
    return acc

acc1,acc2,acc3,acc4,acc5=acc_total(model1),acc_total(model2),acc_total(model3),acc_total(model4),acc_total(model5)
print((acc1+acc2+acc3+acc4+acc5)/5)
print(acc1)
print(acc2)
print(acc3)
print(acc4)
print(acc5)

x=np.linspace(1,len(accc),len(accc))
plt.plot(x,accc,c=color,label=label)
plt.xlabel("迭代次数")
plt.ylabel("准确性")
plt.legend(loc=4) 
plt.title(title)
plt.savefig(save_png,dpi=500,bbox_inches='tight')
plt.show()

# model1.save_weights('wine_nn_advance1')
# model2.save_weights('wine_nn_advance2')
# model3.save_weights('wine_nn_advance3')
# model4.save_weights('wine_nn_advance4')
# model5.save_weights('wine_nn_advance5')
model1.load_weights('wine_nn_advance1')
model2.load_weights('wine_nn_advance2')
model3.load_weights('wine_nn_advance3')
model4.load_weights('wine_nn_advance4')
model5.load_weights('wine_nn_advance5')



#making confusion matrix
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
%matplotlib inline 
import matplotlib
import matplotlib.pyplot as plt
import itertools
# 下载--解压--移动字体文件
!wget "https://www.wfonts.com/download/data/2014/06/01/simhei/simhei.zip"
!unzip "simhei.zip"
!rm "simhei.zip"
!mv SimHei.ttf /usr/share/fonts/truetype/
zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/SimHei.ttf')
plt.rcParams['axes.unicode_minus'] = False

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.autumn_r):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,fontproperties=zhfont, rotation=45)
    plt.yticks(tick_marks, classes,fontproperties=zhfont)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('nn_confusion_matrix.png',dpi=500,bbox_inches='tight')

attribute,label=load_data(filename)
attribute,label=preprocess(attribute,label)
prediction=model1(attribute)
prediction=tf.nn.sigmoid(prediction)
prediction = tf.nn.softmax(prediction, axis=1)
prediction = tf.argmax(prediction, axis=1)
prediction = tf.cast(prediction, dtype=tf.int32)

cnf_matrix = confusion_matrix(label,prediction,  labels=[0,1,2])
np.set_printoptions(precision=2)

print (classification_report(label,prediction))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["琴酒","雪莉","贝尔摩德"],normalize= False,  title='Confusion matrix')

from sklearn.metrics import f1_score
#类别平均
print(f1_score(label,prediction, average='macro'))
#样本平均
print(f1_score(label,prediction, average='micro'))
