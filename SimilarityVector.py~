import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from gensim import corpora, models, similarities 
import re
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score
from gensim.models import word2vec
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")
from multiprocessing.dummy import Pool as ThreadPool 
import itertools
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from gensim.models import Phrases

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
RELATED = LABELS[0:3]

def getF1(gLabels,tLabels):
	return (f1_score(gLabels,tLabels))

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

def getWordVector(word):
	return word_vecs[word]

def getWordFromID(wordID):
	return dictionary[wordID]

def getCosineSimilarity(vec1,vec2):
	return cosine_similarity(vec1,vec2)

def getmissingW2V():
	return (np.random.uniform(-0.25,0.25,300))

def getSimilarityVectors(lst):
	bodyID = lst[1]
	stanceID = lst[0]
	#print i
	simVector = np.zeros(len(dictionary))
	try:
	    for wordID,value in CorporaFNCbody[bodyDictString[bodyID]]:
		    #if wordID in PresentWordIDS.values():
		sims = [getCosineSimilarity(word_vecs[getWordFromID(wordID)],word_vecs[stanceWord]) for stanceWord in stanceDict[stanceID]]
			#print sims
		try:
			simVector[wordID] = np.max(sims)
		except ValueError:
			simVector[wordID] = 0.0
	except KeyError:
		print lst[0]
	return simVector


def getlabelEncoding(lst):
	fncClass = lst[2]
	if fncClass == 'agree':
		return 0
	elif fncClass == 'disagree':
		return 1
	elif fncClass == 'discuss':
		return 2

def getWordVectors(word):
	try:
		return 	


#read the stances and body

TrainStances = pd.read_csv('data/train_stances.csv')
TrainBodies = pd.read_csv('data/train_bodies.csv') 

#define stopWords and the lemmatizer
stop = stopwords.words('english') 
lemmatizer = WordNetLemmatizer()

#process the body text
bodyIds = TrainBodies['Body ID']
bodyText = TrainBodies['articleBody'] 

bodyText = np.array(bodyText)
bodyTextOrig = bodyText
bodyIds = np.array(bodyIds)
bodyText = [line.lower().split() for line in bodyText]
bodyText = [[lemmatizer.lemmatize(re.sub('[^a-zA-Z0-9]', '',word)) for word in text if word not in stop] for text in bodyText]
bigram = Phrases(bodyText, min_count=1, threshold=2)
bodyText = [bigram(text) for text in bodyText]
#create the dictionary for article bodies

bodyDict = dict(zip(bodyIds,bodyText))
bodyDictOrig = dict(zip(bodyIds,bodyTextOrig))
#create the dictionary of words

dictionary = corpora.Dictionary(bodyDict.values())
CorporaFNCbody = [dictionary.doc2bow(sent) for sent in bodyDict.values()]  
tfIdf = models.TfidfModel(CorporaFNCbody)

FncTfidf = tfIdf[CorporaFNCbody]
bodyDictString = dict(zip(bodyDict.keys(),range(0,len(bodyDict.values())))) 
#define LSI and number of topics (k)
lsi = models.LsiModel(FncTfidf, id2word=dictionary, num_topics=150)
#LSI matrix for FNC body corpus
FncLsi = lsi[FncTfidf]
#create the stance corpora 

sents = TrainStances['Headline']

sents = [sent.lower().split() for sent in sents]
#sentsOrig = sents
sents = [[lemmatizer.lemmatize(re.sub('[^a-zA-Z0-9]', '', w)) for w in sent if w not in stop] for sent in sents]
sents = [bigram[text] for text in sents]

stanceDict = dict(zip(range(0,len(np.unique(sents))),np.unique(sents)))
#stanceDictOrig = dict(zip(range(0,len(np.unique(sentsOrig))),np.unique(sentsOrig)))
#also define the reverse stance dictionary
revStanceDict = { str(v): k for k, v in stanceDict.iteritems()}

stanceCorpora = [dictionary.doc2bow(t) for t in stanceDict.values()]
#print ("a sample stance corpora")
#print(stanceCorpora[0])

stanceTfidf = tfIdf[stanceCorpora]
#computing the LSI for stances
stanceLsi = lsi[stanceTfidf]
#create a list from stance and article body combination called miniList

TrainStances = np.array(TrainStances)
TrainStances[:,0] = [a.lower().split() for a in TrainStances[:,0]]
TrainStances[:,0] = [[lemmatizer.lemmatize(re.sub('[^a-zA-Z0-9]', '', w)) for w in sent if w not in stop] for sent in TrainStances[:,0]]
miniList = []
for i in range(0,len(TrainStances)):
	miniList.append([(revStanceDict[str(TrainStances[i][0])]),TrainStances[i][1],TrainStances[i][2]])


word_vecs = np.load('word2vecDict.npy').item()

CorpusWordVectors = word2vec.Word2Vec(bodyDict.values(),size=300, window=5, min_count=4, workers=4,sg=1)
#stanceDict = corpora.Dictionary(stanceDict.values())
#dictionary.merge_with(stanceDict)
for word in dictionary.values():
    if word not in word_vecs.keys():
 	if len(word.split('_')) > 1:
	     wordVector = np.add(word_vecs
		try:
		    word_vecs[word] = CorpusWordVectors[word]
		except KeyError:
		    word_vecs[word] = getmissingW2V()
print (len(word_vecs))

#X_train,Y_train = [(getSimilarityVectors(stanceID,bodyID),labelEncoding(relation)) for stanceID,bodyID,relation in miniList if relation!='unrelated' ]
#X_train = []
#Y_train = []
#for stanceID,bodyID,relation in miniList:
#	X_train.append(getSimilarityVectors(stanceID,bodyID))
#	Y_train.append(labelEncoding(relation))
#	print len(Y_train)
RelatedList = [lst for lst in miniList if lst[2] != 'unrelated']
#pool = ThreadPool(4) 
#X_train,Y_train = pool.map(getSimilarityVectors,RelatedList)
X_train = [getSimilarityVectors(lst) for lst in (RelatedList)]
Y_train = [getlabelEncoding(lst) for lst in RelatedList]
X_train = np.array(X_train)
Y_train = np_utls.to_categorical(Y_train)
#Train the neural network usin tensorflow	

input_num_units = len(dictionary)
hidden_num_units = 500
output_num_units = 3

x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

epochs = 20
batch_size = 128
learning_rate = 0.01

#weight 1: input to hidded, should be a matrix of size input layers x hidden layers
#weight 2: hidden layer x num of outputs
weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

#define the cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.initialize_all_variables()

split_size = int(X_train.shape[0]*0.8)
train_x, val_x = X_train[:split_size], X_train[split_size:]
train_y, val_y = Y_train[:split_size], Y_train[split_size:]

seed = 666
rng = np.random.RandomState(seed)
#define the batch creator
def batch_creator(batch_size, dataset_length, dataset):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval(dataset_name + '_x')[[batch_mask]]

    
    batch_y = eval(dataset_name+'_y')[[batch_mask]]
        
    return batch_x, batch_y

#Train and test the model

with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    
    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(X_train.shape[0]/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            
            avg_cost += c / total_batch
            
        print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)
    
    print "\nTraining complete!"
    
    
    # find predictions on val set
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print "Validation Accuracy:", accuracy.eval({x: val_x, y: val_y})


