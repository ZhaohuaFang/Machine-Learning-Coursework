import os
from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/My Drive/ML"   

os.chdir(path)
os.listdir(path)

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from os import system
import graphviz 
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import scipy.optimize as opt
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
                          cmap=plt.cm.Blues): 
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
    plt.savefig('tree_confusion_matrix.png',dpi=500,bbox_inches='tight')

data=pd.read_csv('/content/drive/My Drive/ML/wine.csv')
X=data[['0','1','2','3','4','5','6','7','8','9','10','11','12']].values
Y=data['13']
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2,random_state=4)
clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)#返回预测的准确accuracy
print(score)
feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','OD280/OD315稀释葡萄酒','脯氨酸']

dot_data = tree.export_graphviz(clf,out_file = None,feature_names= feature_name,class_names=["琴酒","雪莉","贝尔摩德"],filled=True,rounded=True) 
graph = graphviz.Source(dot_data)
graph
# graph.format = 'png'
# graph.render("test",view=True)
#graph.view()
# system("dot -Tpng dtree2.png")

clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=30 ,splitter="random") 
clf = clf.fit(Xtrain, Ytrain) 
score = clf.score(Xtest, Ytest)
score

import graphviz 
dot_data = tree.export_graphviz(clf,feature_names= feature_name,class_names=["琴酒","雪莉","贝尔摩德"],filled=True,rounded=True ) 
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("test",view=True)  

#我们的树对训练集的拟合程度如何？
score_train = clf.score(Xtrain, Ytrain)
score_train
