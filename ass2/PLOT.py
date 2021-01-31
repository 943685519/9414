import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys
import csv 
import numpy as np
import re


def read_file(file):
    list_file = []
    with open(file,'r',encoding = 'UTF-8') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        for row in reader:
            list_file.append(row)
        arr_file = np.array(list_file)

    return arr_file

file = 'dataset.tsv'
arr_file=read_file(file)
cla = list(arr_file[:,-1])


x = ['positive','negative','neutral']
y = [cla.count('positive'),cla.count('negative'),cla.count('neutral')]
plt.figure(figsize=(10,10))
plt.bar(x,y,color = '#9999ff',width = 0.5)       
plt.grid(axis='x', which='major')
plt.title('classes frequency distribution over 5000 tweets',fontsize=25) 
plt.xlabel('sentiment classes',fontsize=25) 
plt.ylabel('frequency distribution',fontsize=25) 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
for a,b in zip(x,y):
    plt.text(a, b,'%d'%b, ha = 'center',va = 'bottom',fontsize=20)
plt.show() 

def test_MultinomialNB_alpha(*data):
    '''
    测试 MultinomialNB 的预测性能随 alpha 参数的影响
    '''
    X_train,X_test,y_train,y_test=data
    alphas=np.logspace(-2,5,num=200)
    test_scores=[]
    for alpha in alphas:
        clf = MultinomialNB(alpha = alpha)
        model = clf.fit(X_train_bag_of_words, y_train)
        predicted_y = predict_and_test(model, X_test_bag_of_words)
        acc_MNB = accuracy_score(y_test, predicted_y)
        test_scores.append(acc_MNB)
    max_indx=max(test_scores)#max value index
    n=test_scores.index(max_indx)

    ## 绘图
    fig=plt.figure(figsize=(10,6))
    ax=fig.subplots()
    ax.plot(alphas,test_scores,label="Testing Score")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel('accuracy')
    ax.set_ylim(0,1.0)
    ax.set_title("MultinomialNB")
    ax.set_xscale("log")


    plt.show()
    return n
    
# 调用 test_MultinomialNB_alpha    
a = test_MultinomialNB_alpha(X_train,X_test,y_train,y_test)

