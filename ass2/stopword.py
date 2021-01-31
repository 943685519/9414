import sys
import csv 
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize 
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



def stopword(sdata):
    afdata = []
    stop_words = set(stopwords.words('english')) 
    for i in range(len(sdata)):
        sentence = sdata[i]
        word_tokens = sentence.split(' ')
        filtered_w = [w for w in word_tokens if w not in stop_words]
        sc = " ".join(filtered_w)
        afdata.append(sc)
    return afdata


def Pstem(sdata):
    afdata = []
    porter_stemmer = PorterStemmer()
    for i in range(len(sdata)):
        sentence = sdata[i]
        word_tokens = sentence.split(' ')
        swords = [porter_stemmer.stem(w) for w in word_tokens]
        sc = " ".join(w for w in swords)
        afdata.append(sc)
    return afdata     
   
    

def predict_and_test(model, X_test_bag_of_words):
    predicted_y = model.predict(X_test_bag_of_words)
    print(classification_report(y_test, predicted_y,zero_division=0))
    return predicted_y 



file = 'dataset.tsv'
arr_file=read_file(file)
id_ = arr_file[:,0]
sdata = pre_sentence(arr_file)
sdata2 = Pstem(sdata)
cla = arr_file[:,-1]

X_train = sdata2[:4000]
X_test = sdata2[4000:]
y_train = cla[:4000]
y_test = cla[4000:]

count = CountVectorizer(token_pattern='[#@_$%\w\d]{2,}',max_features = 1000)
X_train_bag_of_words = count.fit_transform(X_train)

X_test_bag_of_words = count.transform(X_test)




print("----dt")
clf = tree.DecisionTreeClassifier(min_samples_leaf=int(0.01*len(X_train)),criterion='entropy',random_state=0)
model = clf.fit(X_train_bag_of_words, y_train)
predicted_y  = predict_and_test(model, X_test_bag_of_words)

print("----bnb")
clf = BernoulliNB()
model = clf.fit(X_train_bag_of_words, y_train)
predicted_y = model.predict(X_test_bag_of_words)
pre_prob = model.predict_proba(X_test_bag_of_words)
print(classification_report(y_test, predicted_y,zero_division=0))

print("----mnb")
clf = MultinomialNB()
model = clf.fit(X_train_bag_of_words, y_train)
predicted_y = predict_and_test(model, X_test_bag_of_words)



