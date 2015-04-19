from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
from scipy.stats import pearsonr

import re
import time
import codecs
import numpy as np
import matplotlib.pyplot as plt



def read_from_doc(path):
	return [line.strip().split(' ') for line in codecs.open(path).readlines()]



def cross_valid(X,Y,n_fold):
	clf = Ridge(alpha=1.0)
	total_mean_square = 0
	total_coef = 0
	Y_np = np.array(Y)
	n_samples, n_features = len(X), len(X[0])
	kf_Y = cross_validation.KFold(n_samples, n_fold)
	index = []
	preds = []
	truths = []
	for train_index, test_index in kf_Y:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y_np[train_index], Y_np[test_index]
		

		clf.fit(X_train,y_train)
		y_pred = clf.predict(X_test)
		index += test_index.tolist()
		preds += map(lambda x: 1 if x > 0.5 else 0 ,y_pred.tolist())
		truths += y_test.tolist()
		#print "predict:",map(lambda x: 1 if x > 0.5 else 0,y_pred)
		#print "original:",y_test

		total_mean_square += mean_squared_error(y_test,y_pred) 
		total_coef += clf.coef_
	
		#print 'Coefficient of the prediction (pearsonr): ' , pearsonr(y_pred,y_test) 
	print 'All Coefficient of the prediction (pearsonr): ' , pearsonr(truths,preds) 
	print 'Average mean squared error is: ' , total_mean_square / n_fold

	diff_count = sum([abs(truth - pred) for truth, pred in zip(truths, preds)])
	acc =  100-1.* diff_count/len(truths)*100
	print 'prediction accuracy is %f'%(acc)
	return [total_coef, index , preds]

def scatter_plot(index,name_list,pred,test,start = None,end = None):
	if start is None:
		start = 0
	if end is None:
		end = len(index)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	sub_name_list = [name_list[i] for i in index[start:end]]
	sub_pred = [pred[i] for i in index[start:end]]
	sub_test = test[start:end]

	# error is defined by: predicted_value - test_value_from_goodguide
	sub_error = [sub_pred_i - sub_test_i for sub_pred_i,sub_test_i in zip(sub_pred,sub_test)] 
	ax.scatter(sub_test, sub_pred)
	for i, txt in enumerate(sub_name_list):
		ax.annotate(txt, (sub_test[i],sub_pred[i]))
	plt.xlabel('True Values From Original Set')
	plt.ylabel('Error')	
	return plt

def top_weighted_terms(vocabulary,coef,top_num):
#def top_weighted_terms(vocabulary,ind,coef,top_num):
	most_weighted_term = []
	sorted_indices = np.argpartition(coef,-top_num)[-top_num:]
	result = []
	#print len(sorted_indices)
	#print sorted_indices
	for sorted_indice in sorted_indices:
		#most_weighted_term.append((vocabulary.keys()[vocabulary.values().index(ind[sorted_indice])], coef[sorted_indice]))
		most_weighted_term.append((vocabulary.keys()[vocabulary.values().index(sorted_indice)], coef[sorted_indice]))
	most_weighted_term = sorted(most_weighted_term , key=lambda term_score: term_score[1],reverse = True)

	print 'Term |',' Score'	



	for (k,v ) in  most_weighted_term:
		print k.encode('utf-8'),'|',v
		result.append(k.encode('utf-8'))

	return result	

def process(X,Y):
	N_FOLD = 10
	RETURN_TERM_NUM = 50
	print 'Reading data...'
	start_time = time.time()


	tags = []
	Y = Y
	account_name= []

	print 'Reading data done.'
	print("--- %f seconds ---" % (time.time()-start_time))
	print 

	print 'Pre-processing data...'
	start_time = time.time()
	#Personal Care Health

	for user in X:
		tags.append(' '.join(user[1:]))
		account_name.append(user[0])

	end_time = time.time()
	exe_time = end_time-start_time	

	print 'Pre-processing data done.'
	print("--- %f seconds ---" % (time.time() - start_time))
	print 

	#Fit transform
	print 'Performing fit_transform'
	start_time = time.time()

	countVec = CountVectorizer( min_df = 10, binary = True,ngram_range=(3, 3))
	tfidfTrans  = TfidfTransformer() 


	data_countVec = countVec.fit_transform(tags) # countVector

	data_tfidf = tfidfTrans.fit_transform(data_countVec).toarray() # convert term count matrix to term tf-idf matrix
	
	#print len(data_tfidf[0])
	
	"""
	SKB = SelectKBest(f_regression,100)
	data_tfidf = SKB.fit_transform(data_tfidf,Y)
	#returned feature index
	feature_index = SKB.get_support([True])
	"""
	
	print 'Performing fit_transform done'

	print("--- %f seconds ---" % (time.time()-start_time))
	print 

	print 'Performing Regression model...'
	start_time = time.time()
	
	crossV = cross_valid(data_tfidf,Y,N_FOLD)
	#plot = scatter_plot(crossV[1],account_name,np.array(crossV[2]),np.array(Y),0,50)
	print '\'\'\'\'\'\''

	print 'Performing Regression model done.'
	print("--- %s seconds ---" % (time.time() - start_time))
	print

	print 'Returnning top weighted terms...'
	start_time = time.time()
	vocabulary = countVec.vocabulary_
	print vocabulary.keys()[0],vocabulary[vocabulary.keys()[0]]
	#print crossV[0][0:100]
	print '\'\'\'\'\'\''
	top_term = top_weighted_terms(vocabulary,crossV[0],RETURN_TERM_NUM)
	print '\'\'\'\'\'\''
	print

	print 'Returnning top weighted terms done.'
	print("--- %f seconds ---" % (time.time()-start_time))
	
	return top_term

def remove_term(X , term_list):
	result = list()
	for i in range(len(X)):
		temp = ' '.join(X[i]).lower()
		for term in term_list:
			if term in temp:
				#print "term %s is in doc %d"%(term , i+1)
				if term in temp:
					print temp
				temp = temp.replace(' '+term,'')

			else:
				#print "term %s is NOT in doc %d"%(term , i+1)
				pass
		result.append(temp.split(' '))
	return result


if __name__ == '__main__':
	path = 'auther_conversation_with_label.txt'
	#Y_file_path_Personal_Care_Health = '../data/tweco/goodguide2/merged_Y_with_brands_Personal_Care_Health_2.txt'
	i=1
	f_X = list()
	f_Y = list()
	term_to_remove = []
	iters = 5

	with open(path,'r') as f:
		for line in f:
			lineStrip = map(lambda x:x.strip('.?!'),line.strip().split(' '))
			#print i 
			i+=1
			f_X.append(lineStrip[:-1])
			f_Y.append(int(lineStrip[-1:][0]))
	f.close()

	for i in range(iters):
		print '\nPerforming the %dth iteration.'%(i+1)
		f_X = remove_term(f_X,term_to_remove)
		term_to_remove = process(f_X, f_Y)
		print term_to_remove
	#plt.show()
	
	
	