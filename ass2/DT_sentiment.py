import sys
import csv 
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

#traindata = sys.argv[1]
#testdata = sys.argv[2]

def read_file(file):
    list_file = []
    with open(file,'r',encoding = 'UTF-8') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        for row in reader:
            list_file.append(row)
        arr_file = np.array(list_file)
    return arr_file

def pre_sentence(arr_file):
    sdata = arr_file[:,1]
    afdata = []
    url_reg = '[a-zA-z]+://[^\s]*'
    junk_reg = '[#@_$%\s\
    ]'
    for i in range(len(sdata)):
        s1 = re.sub(url_reg, ' ', sdata[i])
        s2 = re.findall(junk_reg,s1,re.S) 
        s2 = "".join(s2)
        afdata.append(s2)
    return afdata




def predict_and_test(model, X_test_bag_of_words):
    predicted_y = model.predict(X_test_bag_of_words)
    print(classification_report(y_test, predicted_y,zero_division=0))
    return predicted_y 




arr_file1=read_file('trainset.tsv')
arr_file2=read_file('testset.tsv')
sdata1 = pre_sentence(arr_file1)
cla1 = arr_file1[:,-1]

sdata2 = pre_sentence(arr_file2)
cla2 = arr_file2[:,-1]
id_ =  arr_file2[:,0]

X_train = sdata1
X_test = sdata2
y_train = cla1
y_test = cla2

count = CountVectorizer(token_pattern='[#@_$%\w\d]{2,}',lowercase = False)
X_train_bag_of_words = count.fit_transform(X_train)

X_test_bag_of_words = count.transform(X_test)

print("----dt")
clf = tree.DecisionTreeClassifier(min_samples_leaf=int(0.01*len(X_train)),criterion='entropy',random_state=0)
model = clf.fit(X_train_bag_of_words, y_train)
predicted_y  = predict_and_test(model, X_test_bag_of_words)
for i in range(len(X_test)):
    print(id_[:][i],predicted_y[i])
    




    
    
    
    
    
    
    