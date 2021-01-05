clf=tree.DecisionTreeClassifier(criterion="entropy",random_state=30,splitter="random",max_depth=3,min_samples_leaf=5,min_samples_split=5)
clf = clf.fit(Xtrain, Ytrain)
dot_data = tree.export_graphviz(clf,feature_names= feature_name,class_names=["琴酒","雪莉","贝尔摩德"],filled=True,rounded=True)
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("test2",view=True) 

score = clf.score(Xtest, Ytest)
score

Yhat = clf.predict(Xtest)
cnf_matrix = confusion_matrix(Ytest, Yhat, labels=[1,2,3])
np.set_printoptions(precision=2)

print (classification_report(Ytest, Yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["琴酒","雪莉","贝尔摩德"],normalize= False,  title='Confusion matrix')

# 下载--解压--移动字体文件
!wget "https://www.wfonts.com/download/data/2014/06/01/simhei/simhei.zip"
!unzip "simhei.zip"
!rm "simhei.zip"
!mv SimHei.ttf /usr/share/fonts/truetype/
zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/SimHei.ttf')
plt.rcParams['axes.unicode_minus'] = False
import seaborn
test = []
for i in range(20):
    clf = tree.DecisionTreeClassifier(max_depth=i+1,criterion="entropy",random_state=30,splitter="random")
    clf = clf.fit(Xtrain, Ytrain)
    score = clf.score(Xtest, Ytest)
    print(score)
    test.append(score)
plt.plot(range(1,21),test,color=seaborn.xkcd_rgb['medium blue'],label="max_depth")
plt.legend()
print(max(test))
plt.xlabel('最大深度',fontproperties=zhfont,)
plt.ylabel('准确率',fontproperties=zhfont,)
plt.savefig('depth.png',dpi=500,bbox_inches='tight')
plt.show()

clf=tree.DecisionTreeClassifier(criterion="entropy",random_state=30,splitter="random",max_depth=3)
clf = clf.fit(Xtrain, Ytrain)
dot_data = tree.export_graphviz(clf,feature_names= feature_name,class_names=["琴酒","雪莉","贝尔摩德"],filled=True,rounded=True)
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("test2",view=True) 
score = clf.score(Xtest, Ytest)
print(score)
