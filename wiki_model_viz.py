import logging, gensim
from os import path
import sys
import wordcloud
import numpy as np


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

id2word = gensim.corpora.Dictionary.load_from_text('data/wiki_en_output_wordids.txt')

mm = gensim.corpora.MmCorpus('data/wiki_en_output_tfidf.mm')

model = gensim.models.ldamodel.LdaModel(corpus=mm,id2word=id2word,num_topics=100,update_every=1,chunksize=10000,passes=1)

# saving model so it can be reused later if needed.
model.save('wiki_lda_100_topics.pkl')

# to load:
#model = gensim.models.ldamodel.LdaModel.load('wiki_lda.pkl')

topics = []
for doc in mm:
	topics.append(model[doc])

#------------
# to get the average number of topics in a document
lens = np.array([len(t) for t in topics])
print np.mean(lens)
#------------

# -------------- WORD CLOUD -------------------
counts = np.zeros(100)
for doc_top in topics:
	for ti,_ in doc_top:
		counts[ti] += 1

# most talked about topics
words_max = model.show_topic(counts.argmax(), 50)

# least talked about topics
words_min = model.show_topic(counts.argmin(), 50)

wf_max = []
wlist_max = []
for i,j in words_max:
	wlist_max.append(j)
for i in range(50):
	wf_max.append((wlist_max[i],counts[i]))

wf_min = []
wlist_min = []
for i,j in words_min:
	wlist_min.append(j)
for i in range(50):
	wf_min.append((wlist_min[i],counts[i+50]))

d = path.dirname(__file__)

elements_max = wordcloud.fit_words(wf_max)
wordcloud.draw(elements_max, path.join(d, 'top50.png'),scale=3)

elements_min = wordcloud.fit_words(wf_min)
wordcloud.draw(elements_min, path.join(d, 'bottom50.png'),scale=3)