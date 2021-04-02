############################################
#Computer Security Final Project - Detecting Malicious URL using Machine Learning algorithm
#Sumbitted by: Abdullah Alshuaibi, Arghadeep Mitra and Vinay Sharma
#############################################

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import Counter
from flask import Flask
import pandas as pd
import numpy as np
import random
import math
import os

app = Flask(__name__)

def entropy(labels):
	l, ava = Counter(labels), float(len(labels))
	return -sum( count/ava * math.log(count/ava, 2) for count in l.values())

########## Defining and Initializing tokens and spliiting the various words of a url which are common and found on ervery website url.############
def fetchtkns(input):
	slashtokens = str(input.encode('utf-8')).split('/')	#Splitting the slash and then fetching tokens. 
	everyTokens = []
	for i in slashtokens:
		tokens = str(i).split('-')	#Splitting the dash and then fetching tokens.
		dottokens = []
		for j in range(0,len(tokens)):
			temporarytokens = str(tokens[j]).split('.')	#Splitting the dot and then fetching tokens.
			dottokens = dottokens + temporarytokens
		everyTokens = everyTokens + tokens + dottokens
	everyTokens = list(set(everyTokens))	#Deleting the unnecessary tokens whcih are not needed.
	if 'com' in everyTokens:
		everyTokens.remove('com')	#".com" is one of the basic feature of a url i.e. why it is discarded. 
	return everyTokens

#### Loading and reading the dataset of the website's URL's. Vectorizer and Customized tokenizer is used. #### 
def ml_algo():
	everyurls = './data/data.csv'	#Dataset of Good and Bad Url's.
	everycsvurls = pd.read_csv(everyurls,',',error_bad_lines=False)	#Reading dataset file.
	everydataurls = pd.DataFrame(everycsvurls)	#Change Url's into dataframe.

	everydataurls = np.array(everydataurls)	#Transforming the URL's into an array.
	random.shuffle(everydataurls)	#Shuffling of URL's.

	q = [d[1] for d in everydataurls]	#Each labels 
	corrsurls = [d[0] for d in everydataurls]	#Each website url's defined as a label in the dataset. Label is defined for URL's as Good & Bad.
	vectorizer = TfidfVectorizer(tokenizer=fetchtkns)	#Fetching vectors for every website's url. Customized tokenizer is used.
	P = vectorizer.fit_transform(corrsurls)	#get the P vector

	P_train, P_test, q_train, q_test = train_test_split(P, q, test_size=0.2, random_state=42)	#Dividing training set and testing set in a 80:20 proportion.
	
	####Initializing logistic regression for classifying the website's URL's.####
	lor = LogisticRegression()
	lor.fit(P_train, q_train)
	print(lor.score(P_test, q_test))	#Display the accuracy.
	return vectorizer, lor

@app.route('/<path:path>')
def show_index(path):
	P_predict = []
	P_predict.append(str(path))
	P_predict = vectorizer.transform(P_predict)
	q_Predict = lor.predict(P_predict)
	return '''The URL which you have typed %s is : %s ''' % (str(path), str(q_Predict))
port = os.getenv('VCAP_APP_PORT', 1235)
if __name__ == "__main__":
	vectorizer, lor  = ml_algo()
	app.run(host='0.0.0.0',port=int(port), debug=True)



