import sys
import csv 
import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#filedata = sys.argv[1]
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
    junk_reg = '[#@_$%\s\w\d]'
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

file = 'dataset.tsv'
arr_file=read_file(file)
id_ = arr_file[:,0]
sdata = pre_sentence(arr_file)
cla = arr_file[:,-1]


X_train = sdata[:4000]
X_test = sdata[4000:]
y_train = cla[:4000]
y_test = cla[4000:]


count = CountVectorizer(token_pattern='[#@_$%\w\d]{2,}',max_features = 1000)
X_train_bag_of_words = count.fit_transform(X_train)

X_test_bag_of_words = count.transform(X_test)

print("----mnb")
clf = MultinomialNB(alpha = 2.6738416158399465)
model = clf.fit(X_train_bag_of_words, y_train)
predicted_y = predict_and_test(model, X_test_bag_of_words)
#for i in range(len(X_test)):
    #print(id_[4000:][i],predicted_y[i])


acc_MNB = accuracy_score(y_test, predicted_y)
print('Overall accuracy of MNB model:', acc_MNB)



