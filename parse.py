from bs4 import BeautifulSoup
from lxml import etree
from collections import defaultdict
import string
import re
	
#soup = BeautifulSoup(open('raw.xml'))
path = []
auther_conversation = defaultdict(list)
conver_pervert = defaultdict(list)

positive_id = set() # List of ID that are prediater.

i = 1

context = etree.parse('raw2.xml')
conversations = context.getroot()


print "Creating conversations with labled message."
with open("problem2.txt", "rw") as f:
	for line in f:
		line_token = line.strip().split('\t')
		conver_pervert[line_token[0]].append(line_token[1])
f.close()
print "Done."


#def remove():
print "Removing labled message."
for conversation in conversations:
	"""
	conversation has attribute id
	ele has: tag and attribute line
	ele has 3 sub nodes: author, time, text.
	"""	
	for ele in conversation:
		if ele.attrib['line'] not in conver_pervert[conversation.attrib['id']]:
			if ele[2].text is not None:
			#	auther_conversation[ele[0].text].append(ele[2].text.encode('utf8').translate(string.maketrans("\n\t\r", "   ")))
				auther_conversation[ele[0].text].append(re.sub('[!@#$&*.,\':;/())?]','',ele[2].text.encode('utf8')))
		else:
			positive_id.add(ele[0].text)
print "Done"

"""
Write auth_dict, positive_id to file.
"""

with open("auther_conversation_2.txt", "w") as f:
	for (key , value) in auther_conversation.iteritems():
		i+=1

		text=key + ' ' + ', '.join(auther_conversation[key]).strip(",.!@#$%?<>()[]-=+*/\'")
		#sentence = [' '
		f.write(text)
		f.write('\n')
		if i%100==0:
			print "Wrote %d lines to file."%i
		#print i
f.close()


with open("predator_id_problem2.txt","w") as f:
	for p_id in list(positive_id):
		f.write(p_id+'\n')
f.close()
