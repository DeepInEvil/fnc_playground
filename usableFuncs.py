from nltk.tag.stanford import StanfordPOSTagger

englishPOStagger = StanfordPOSTagger('/home/deep/StanfordPosTagger/models/english-bidirectional-distsim.tagger','/home/deep/StanfordPosTagger/stanford-postagger.jar')
java_path='/usr/bin/java'
os.environ['JAVAHOME'] = java_path
def updatePosDict(tokens,PosDict):
	tags = englishPOStagger.tag(tokens)
	for token,tag in tags:
		PosDict[tag] += 1

PosAgrees = defaultdict(float)
PosDisagrees = defaultdict(float)
PosDiscuss = defaultdict(float)
NumDiscuss = 0.0
NumAgreed = 0.0
NumDisagreed = 0.0
for i,lst in enumerate(miniList):
	if lst[2] == 'agree':
		NumAgreed += 1.0
		try:
			updatePosDict(bodyDict[lst[1]] , PosAgrees)
		except OSError:
			print (i) 
			
	elif lst[2] == 'disagree':
		NumDisagreed += 1.0
		try:
			updatePosDict(bodyDict[lst[1]] , PosDisagrees)
		except OSError:
			print (i) 
	elif lst[2] == 'discuss':
		NumDiscuss += 1.0
		try:
			updatePosDict(bodyDict[lst[1]] , PosDiscuss)
		
		except OSError:
			print (i) 


LdaTopicAgree = []
LdaTopicDisagree = []
LdaTopicDiscuss = []

for i,lst in enumerate(miniList):
	if lst[2] == 'agree':
		X_train.append(getDistributionsVector(LdaModel.get_document_topics(CorporaFNCbody[bodyDictString[lst[1]]],minimum_probability=0)))
		Y_train.append(0)
	elif lst[2] == 'disagree':
		#X_train.append(getDistributionsVector(LdaModel.get_document_topics(CorporaFNCbody[bodyDictString[lst[1]]],minimum_probability=0)))
		Y_train.append(1)
	elif lst[2] == 'discuss':
		#X_train.append(getDistributionsVector(LdaModel.get_document_topics(CorporaFNCbody[bodyDictString[lst[1]]],minimum_probability=0)))
		Y_train.append(2)


def getDistributionsVector(topicDict):

	DistVector = np.zeros(len(topicDict))
	#print (len(DistVector))
	
	for i in range(0,len(topicDict)):
		DistVector[i] = topicDict[i][1]

	return DistVector


def getDistTopic(distList):
	
	finalVector = np.zeros(100)
	for lst in distList:
		finalVector = np.add(finalVector,getDistributionsVector(lst))

	print (finalVector)
	return np.divide(np.array(finalVector),len(distList))
f_scores = []
for train_index, test_index in kf.split(X_train):
	X_tr, X_te = X_train[train_index], X_train[test_index]
	Y_tr, Y_te = Y_train[train_index], Y_train[test_index]

	model = Sequential()
	model.add(Dense(80, input_dim=100, init="lecun_uniform",activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(3))
	model.add(Activation('softmax'))

	#sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
	#optm = RMSprop(lr=0.004, rho=0.9, epsilon=1e-08, decay=0.0)
	ada = keras.optimizers.Adagrad(lr=0.08, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy', optimizer=ada)

	model.fit(X_tr,Y_tr, batch_size=128 , nb_epoch=20)


	Y_predicted = model.predict(X_te)
	Y_pred = np.zeros(Y_predicted.shape)
	for i,prediction in enumerate(Y_predicted):
		Y_pred[i][np.argmax(prediction)] = 1


	print ("F1 score:"  + str(f1_score(Y_te,Y_pred,average='micro')))
	f_scores.append(f1_score(Y_te,Y_pred,average='micro'))


f_scores = []
for train_index, test_index in kf.split(X_train):
	X_tr, X_te = X_train[train_index], X_train[test_index]
	Y_tr, Y_te = Y_train[train_index], Y_train[test_index]

	model = Sequential()
	model.add(Dense(80, input_dim=100, init="lecun_uniform",activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(3))
	model.add(Activation('softmax'))

	#sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
	#optm = RMSprop(lr=0.004, rho=0.9, epsilon=1e-08, decay=0.0)
	ada = keras.optimizers.Adagrad(lr=0.08, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy', optimizer=ada)

	model.fit(X_tr,Y_tr, batch_size=128 , nb_epoch=20)


	Y_predicted = model.predict(X_te)
	Y_pred = np.zeros(Y_predicted.shape)
	for i,prediction in enumerate(Y_predicted):
		Y_pred[i][np.argmax(prediction)] = 1


	print ("F1 score:"  + str(f1_score(Y_te,Y_pred,average='micro')))
	f_scores.append(f1_score(Y_te,Y_pred,average='micro'))



kl_diverScore = []
for train_index, test_index in kf.split(X_train):
	X_tr, X_te = X_train[train_index], X_train[test_index]
	Y_tr, Y_te = Y_train[train_index], Y_train[test_index]

	model = Sequential()
	model.add(Dense(2000, input_dim=23388, init="lecun_uniform",activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(100))
	model.add(Activation('softmax'))

	#sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
	#optm = RMSprop(lr=0.004, rho=0.9, epsilon=1e-08, decay=0.0)
	ada = keras.optimizers.Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)
	model.compile(loss='kullback_leibler_divergence', optimizer=ada)

	model.fit(X_tr,Y_tr, batch_size=128 , nb_epoch=20)


	Y_predicted = model.predict(X_te)



	print ("kl divergance value"  + str(entropy(Y_te,Y_pred,base=float)))
	kl_diverScore.append(entropy(Y_te,Y_pred,base=float))

bodyTextDup = defaultdict(list)
for lst in miniList:
    if lst[2] != 'unrelated':
         bodyTextDup[lst[1]].append([lst[0],lst[2]])



def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
		print (len(word_vecs))
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    	
    return word_vecs

import numpy as np

# Save
dictionary = {'hello':'world'}
np.save('my_file.npy', dictionary) 

# Load
read_dictionary = np.load('my_file.npy').item()
print(read_dictionary['hello']) # displays "world"

for word in vocab:
if word not in word_vecs and vocab[word] >= min_df:
    word_vecs[word] = np.random.uniform(-0.25,0.25,k)  
