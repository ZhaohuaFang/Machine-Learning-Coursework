import os
from google.colab import drive

drive.mount('/content/drive')

path = "/content/drive/My Drive/ML"   

os.chdir(path)
os.listdir(path)

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.model_selection import train_test_split
%matplotlib inline 
import matplotlib
import matplotlib.pyplot as plt

# 下载--解压--移动字体文件
!wget "https://www.wfonts.com/download/data/2014/06/01/simhei/simhei.zip"
!unzip "simhei.zip"
!rm "simhei.zip"
!mv SimHei.ttf /usr/share/fonts/truetype/
zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/SimHei.ttf')
plt.rcParams['axes.unicode_minus'] = False


data = pd.read_csv('/content/drive/My Drive/ML/wine.csv')
X=data[['0','1','2','3','4','5','6','7','8','9','10','11','12']].values
Y=data['13']
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
print(X_test.shape)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 
yhat = clf.predict(X_test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.BuPu):
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
    plt.savefig('svm_confusion_matrix.png',dpi=500,bbox_inches='tight')
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,2,3])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["琴酒","雪莉","贝尔摩德"],normalize= False,  title='Confusion matrix')

from sklearn.metrics import f1_score
#类别平均
print(f1_score(y_test, yhat, average='macro')) 
#样本平均
print(f1_score(y_test, yhat, average='micro'))
