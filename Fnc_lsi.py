import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from gensim import corpora, models, similarities 
import re
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score
from gensim.models import Phrases
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
import keras
from collections import Counter
import score
import warnings
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore",category=DeprecationWarning)

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]


def getF1(gLabels,tLabels):
	return (f1_score(gLabels,tLabels,average="macro"))

def clean_str(string, TREC=False):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Every dataset is lower cased except for TREC
	"""
	try:
		string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
		string = re.sub(r"\'s", " \'s", string) 
		string = re.sub(r"\'ve", " \'ve", string) 
		string = re.sub(r"n\'t", " n\'t", string) 
		string = re.sub(r"\'re", " \'re", string) 
		string = re.sub(r"\'d", " \'d", string) 
		string = re.sub(r"\'ll", " \'ll", string) 
		string = re.sub(r",", " , ", string) 
		string = re.sub(r"!", " ! ", string) 
		string = re.sub(r"\(", " \( ", string) 
		string = re.sub(r"\)", " \) ", string) 
		string = re.sub(r"\?", " \? ", string) 
		string = re.sub(r"\s{2,}", " ", string)    
	except Exception:
		print string

	return string.strip().lower().split()
	#get the values for f1 score for related and unrelated articles
def getCosineThreshold(cosineSimMatrix,val):
	generateLabelsonThreshold = []
	generateOrigLabels = []

	for cosineDist, label in cosineSimMatrix:
		if cosineDist > val:
			generateLabelsonThreshold.append(1)
		else:
			generateLabelsonThreshold.append(0)
		if label in RELATED:
			generateOrigLabels.append(1)
		else:
			generateOrigLabels.append(0)

	return getF1(generateLabelsonThreshold,generateOrigLabels)

#read the stances and body

TrainStances = pd.read_csv('data/train_stances.csv')
TrainBodies = pd.read_csv('data/train_bodies.csv') 




#define stopWords and the lemmatizer
stop = ['what','is','the','on','in','where','be','was','to','from','why','i','you','this','that','a','an','into','how','these','we',
'them','if','for','at','by','those','of','he','or','it','will','which','they','them','us','we','whom','here','there','did','have']
lemmatizer = WordNetLemmatizer()

#process the body text
bodyIds = TrainBodies['Body ID']
bodyText = TrainBodies['articleBody'] 

bodyText = np.array(bodyText)
bodyTextOrig = bodyText
bodyIds = np.array(bodyIds)
"""
bodyText = [line.lower().split() for line in bodyText]
bodyText = [[lemmatizer.lemmatize(re.sub('[^a-zA-Z0-9]', ' ',word)).strip() for word in text if word not in stop] for text in bodyText]
"""
bodyText = [clean_str(body) for body in bodyText]
bodyText = [[lemmatizer.lemmatize(re.sub('[^a-zA-Z0-9]', ' ',word)).strip() for word in text if word not in stop] for text in bodyText]
#create the dictionary for article bodies

bodyDict = dict(zip(bodyIds,bodyText))
bodyDictOrig = dict(zip(bodyIds,bodyTextOrig))
bodyDictString = dict(zip(bodyDict.keys(),range(0,len(bodyDict.values())))) 

sents = TrainStances['Headline']
"""
sents = [sent.lower().split() for sent in sents]
#sentsOrig = sents
sents = [[lemmatizer.lemmatize(re.sub('[^a-zA-Z0-9]', ' ', w)).strip() for w in sent if w not in stop] for sent in sents]
"""
sents = [clean_str(sent) for sent in sents]
sents = [[lemmatizer.lemmatize(re.sub('[^a-zA-Z0-9]', ' ', w)).strip() for w in sent if w not in stop] for sent in sents]
stances_unique = np.unique(sents)

stanceDict = dict(zip(range(0,len(np.unique(stances_unique))),np.unique(stances_unique)))
#stanceDictOrig = dict(zip(range(0,len(np.unique(sentsOrig))),np.unique(sentsOrig)))
#also define the reverse stance dictionary
revStanceDict = { str(v): k for k, v in stanceDict.iteritems()}

#create the dictionary of words
#all_text = np.concatenate((bodyText,stances_unique),axis=0)





all_text = bodyDict.values()
bigram = Phrases(all_text,threshold=20,min_count=20)
all_text = [bigram[text] for text in all_text]

dictionary = corpora.Dictionary(all_text)
CorporaFNCbody = [dictionary.doc2bow(sent) for sent in all_text]  
tfIdf = models.TfidfModel(CorporaFNCbody)

FncTfidf = tfIdf[CorporaFNCbody]


#define LSI and number of topics (k)
lsi = models.LsiModel(FncTfidf, id2word=dictionary, num_topics=500)
#LSI matrix for FNC body corpus
FncLsi = lsi[FncTfidf]
#create the stance corpora 


stanceCorpora = [bigram[stance] for stance in stanceDict.values()]
#stanceCorpora = stanceDict.values()
stanceCorpora = [dictionary.doc2bow(t) for t in stanceCorpora]
#print ("a sample stance corpora")
#print(stanceCorpora[0])

stanceTfidf = tfIdf[stanceCorpora]
#computing the LSI for stances
stanceLsi = lsi[stanceTfidf]
#create a list from stance and article body combination called miniList

TrainStances = np.array(TrainStances)
#TrainStances[:,0] = [a.lower().split() for a in TrainStances[:,0]]
TrainStances[:,0] = [clean_str(a) for a in TrainStances[:,0]]
TrainStances[:,0] = [[lemmatizer.lemmatize(re.sub('[^a-zA-Z0-9]', ' ', w)).strip() for w in sent if w not in stop] for sent in TrainStances[:,0]]
#TrainStances[:,0] = [clean_str(a) for a in TrainStances[:,0]]

miniList = []
for i in range(0,len(TrainStances)):
	miniList.append([(revStanceDict[str(TrainStances[i][0])]),TrainStances[i][1],TrainStances[i][2]])




def get_lsi_vec(lsi):
	lsi_vec = np.zeros(500,dtype=np.float32)
	for i in range(0,len(lsi)):
		lsi_vec[i] = lsi[i][1]
	#print lsi_vec.shape
	return lsi_vec

def get_lsi_concatenated(stance_id, body_id):

	stance_vec = get_lsi_vec(stanceLsi[stance_id])
	body_vec = get_lsi_vec(FncLsi[bodyDictString[body_id]])

	return np.hstack((stance_vec,body_vec))



X = [get_lsi_concatenated(a,b) for a,b,c in miniList if c != 'unrelated']
y_orig = [c for a,b,c in miniList if c!= 'unrelated']
X = np.array(X)
y = np.array(pd.get_dummies(y_orig))
y_labels = np.array([np.argmax(a) for a in y])


skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X, y)
"""
for train_index, test_index in skf.split(X, y_labels):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = np.array(pd.get_dummies(y_labels[train_index])), (y_labels[test_index])
	print X_train[0] , y_train[0]	
	#define the neural net model
	model = Sequential()
	model.add(Dense(1000, input_shape=(1500,), init = "lecun_uniform"))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(keras.layers.normalization.BatchNormalization())
	model.add(Dense(500))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(keras.layers.normalization.BatchNormalization())
	model.add(Dense(4))
	model.add(Activation('softmax'))
	ada = keras.optimizers.Adagrad(lr=0.02, epsilon=1e-08, decay=0.0)
	model.compile(optimizer=ada, loss='categorical_crossentropy', metrics=['accuracy'])

	model.fit(X_train, y_train, epochs = 40, batch_size = 128,validation_split=0.01)
	
	print ("Distributions:")
	print Counter(y_labels)
	
	y_predicted = model.predict_classes(X_test)

	print ("F1 scores:" + "\n")
	print getF1(y_test,y_predicted)

	print ("\n"  + "Results from FNC:")
	print (score.report_score([LABELS[e] for e in y_test],[LABELS[e] for e in y_predicted]))

#compute the similarities of all sentences in the document bodies 

#sim = similarities.docsim.MatrixSimilarity(FncTfidf)
#Get similarities for LSI
sim = similarities.docsim.MatrixSimilarity(FncLsi)
#stance to body similarities

Stance2BodySim = [sim.get_similarities(stancetf) for stancetf in stanceLsi]

#compute cosine similarity for the milist
cosineSim = [[Stance2BodySim[lst[0]][bodyDictString[lst[1]]] , lst[2]] for lst in miniList]

#plot the histograms for cosine similarities




cosineSimUnrelated = []
for sim in cosineSim:
    if sim[1] == 'unrelated':
        cosineSimUnrelated.append(sim[0])

cosineSimRelated = []
for sim in cosineSim:
    if sim[1] != 'unrelated':
        cosineSimRelated.append(sim[0])


plt.hist(cosineSimRelated)
plt.xlabel("Cosine similarities histogram for related articles")
plt.savefig('plots/histCosineRelated.png')

plt.hist(cosineSimUnrelated)
plt.xlabel("Cosine similarities histogram comparing both related and unrelated articles")
plt.savefig('plots/histCosineBoth.png')

plt.clf()
#plot cosine threshold and f1 score
cosineThreshold = np.arange(0.0,1.0,0.01)
Y = [getCosineThreshold(cosineSim,a) for a in cosineThreshold]
plt.plot(cosineThreshold,Y)
plt.ylabel("F1 score on related/unrelated")
plt.xlabel("cosine threshold value")
plt.savefig('plots/f1Scores.png')

print ("Highest f1 score:" + str(np.max(Y)) + " for cosine threshold value of:"  + str(cosineThreshold[np.argmax(Y)]))

"""



model = Sequential()
model.add(Dense(1200, input_shape=(1000,), init = "glorot_uniform"))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(keras.layers.normalization.BatchNormalization())
model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(keras.layers.normalization.BatchNormalization())
model.add(Dense(3))
model.add(Activation('softmax'))
ada = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
model.compile(optimizer=ada, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs = 30, batch_size = 128,validation_split=0.01)

#get test data
test_stance = pd.read_csv('data/test_stances_unlabeled.csv')
test_body = pd.read_csv('data/test_bodies.csv')

test_bodyIds = test_body['Body ID']
test_bodyText = test_body['articleBody'] 

#process test body
test_bodyText = np.array(test_bodyText)
#bodyTextOrig = bodyText
test_bodyIds = np.array(test_bodyIds)

test_bodyText = [clean_str(body) for body in test_bodyText]
test_bodyText = [[lemmatizer.lemmatize(re.sub('[^a-zA-Z0-9]', ' ',word)).strip() for word in text if word not in stop] for text in test_bodyText]
test_bodyText = [bigram[text] for text in test_bodyText]

test_bodyText = [dictionary.doc2bow(sent,allow_update=True) for sent in test_bodyText]
test_body_tfidf = tfIdf[test_bodyText]
test_body_lsi = lsi[test_body_tfidf]
#create the dictionary for article bodies

test_bodyDict = dict(zip(test_bodyIds,test_body_lsi))


test_sents = test_stance['Headline']
test_sents = [clean_str(sent) for sent in test_sents]
test_sents = [[lemmatizer.lemmatize(re.sub('[^a-zA-Z0-9]', ' ', w)).strip() for w in sent if w not in stop] for sent in test_sents]
test_sents = [bigram[text] for text in test_sents]
test_stances_unique = np.unique(test_sents)
test_stance_dict = dict(zip(range(0,len(test_stances_unique)),test_stances_unique))

test_revStanceDict = { str(v): k for k, v in test_stance_dict.iteritems()}

test_stances_unique = [dictionary.doc2bow(sent) for sent in test_stances_unique]
test_stance_tfidf = tfIdf[test_stances_unique]
test_stance_lsi = lsi[test_stance_tfidf]


test_miniList = []

test_stance = np.array(test_stance)
#TrainStances[:,0] = [a.lower().split() for a in TrainStances[:,0]]
test_stance[:,0] = [clean_str(a) for a in test_stance[:,0]]
test_stance[:,0] = [[lemmatizer.lemmatize(re.sub('[^a-zA-Z0-9]', ' ', w)).strip() for w in sent if w not in stop] for sent in test_stance[:,0]]
test_stance[:,0] = [bigram[text] for text in test_stance[:,0]]

for i in range(0,len(test_stance)):
	test_miniList.append([test_revStanceDict[str(test_stance[i][0])],test_stance[i][1]])

def get_test_lsi_concatenated(stance_id, body_id):

	stance_vec = get_lsi_vec(test_stance_lsi[stance_id])
	body_vec = get_lsi_vec(test_bodyDict[body_id])
	return np.hstack((stance_vec,body_vec))

def get_test_lsi_cosine(stance_id, body_id):

	stance_vec = get_lsi_vec(test_stance_lsi[stance_id])
	body_vec = get_lsi_vec(test_bodyDict[body_id])
	return cosine_similarity(stance_vec,body_vec)

#x_test = [get_test_lsi_concatenated(a,b) for a,b in test_miniList]
#get cosime similarities between lsi vectors of test body and headlines
similarities = [get_test_lsi_cosine(a,b) for a,b in test_miniList]
test_y_label =[]
for sim in similarities:
    if sim > 0.11:
        test_y_label.append('untagged')
    else:
        test_y_label.append('unrelated')

x_test = [(get_test_lsi_concatenated(test_miniList[i][0],test_miniList[i][1])) for i in range(0,len(test_miniList)) if test_y_label[i] == 'untagged']
untagged_articles = [i for i in range(0,len(test_miniList)) if test_y_label[i] == 'untagged']

x_test = np.array(x_test)
predictions = model.predict_classes(x_test)
pred_dictionary = dict(zip(untagged_articles,predictions))
#getting the final labels

for j in untagged_articles:                                        
	test_y_label[j] = RELATED[pred_dictionary[j]]


test_stance_submit = pd.read_csv('data/test_stances_unlabeled.csv')
test_stance_submit['Stance'] = test_y_label

test_stance_submit.to_csv('submission.csv',index=False)


