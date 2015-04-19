from nltk.stem.lancaster import LancasterStemmer
from collections import defaultdict

st = LancasterStemmer()
i = 1
predator = []
with open('predator_id_problem2.txt' ,'r') as f_p:
	for line in f_p:
		predator.append(line.strip('\n'))
		#print line

length = []
j = 0
k = 0
with open('auther_conversation_with_label.txt','w') as X:
	with open('auther_conversation_2.txt', 'r') as auther_conversation_2:
		for line in auther_conversation_2:
			if i % 1000 == 0:
				#print 'processing line %d'%i
				pass
			i+=1
			#lineSplit = map(lambda x:st.stem(x.decode('utf8')) ,line.strip().split(' '))
			#lineSplit = [x.encode('utf8') for x in lineSplit]
			lineSplit = line.strip('\n').split(' ')
			if lineSplit[0] in predator:
				X.write(lineSplit[0]+' '+' '.join(lineSplit[1:])+' '+'1'+'\n')
				#print "Length of predator: %d"%len(lineSplit)
				length.append(len(lineSplit))
				k+=1
			elif len(lineSplit)>=1000 and len(lineSplit)<=6000:  # Only add negative set with document length is greater than 1000
				X.write(lineSplit[0]+' '+' '.join(lineSplit[1:])+' '+'0'+'\n')
				j += 1
				k+=1
				length.append(len(lineSplit))
				print "%dth legnth of non-predator is:%d"%(j,len(lineSplit))
	auther_conversation_2.close()
X.close()

#print "avg length:%f"%1.*sum(length)/len(length)
print "min:%d"%min(length)
print "max:%d"%max(length)
print "total number of lines is %d"%k
