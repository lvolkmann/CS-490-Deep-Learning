from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from bs4 import BeautifulSoup
from bs4.element import Comment
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
import urllib.request as req

# PT I

categories = ['soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, categories=categories)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True, categories=categories)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print("MultinomialNB: " + str(score))

# TO SVC

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = svm.SVC()
clf.fit(X_train_tfidf, twenty_train.target)

X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print("SVC: " + str(score))

# To Bigram
tfidf_Vect = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print("MultinomialNB + Bigram: " + str(score))

# with stop words

tfidf_Vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print("MultinomialNB + Bigram + Stopword: " + str(score))

# # TO KNN
# from sklearn kn
# tfidf_Vect = TfidfVectorizer()
# X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# # print(tfidf_Vect.vocabulary_)
# clf = MultinomialNB()
# clf.fit(X_train_tfidf, twenty_train.target)
#
# twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
# X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
#
# predicted = clf.predict(X_test_tfidf)
#
# score = metrics.accuracy_score(twenty_test.target, predicted)
# print(score)

# PT II
# Credit:https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

# Credit:https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text
def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)


url = "https://en.wikipedia.org/wiki/Google"

html = req.urlopen(url).read()
text = text_from_html(html)

# fh = open("input.txt", 'w', encoding="utf-8")
# fh.write(text)

# tokenize
print("SENTENCE TOKENS:")
sentence_tokens = nltk.sent_tokenize(text)
print(str(sentence_tokens[:5]) + " ... " + " ... and {} others.".format(len(sentence_tokens[5:])))
print()

print("WORD TOKENS:")
word_tokens = nltk.word_tokenize(text)
print(str(word_tokens[:5]) + " ... and {} others.".format(len(word_tokens[5:])))
print()

# POS
print("PARTS OF SPEECH:")
part_of_speech_tags = nltk.pos_tag(word_tokens)
print(str(part_of_speech_tags[:5]) + " ... and {} others.".format(len(part_of_speech_tags[5:])))
print()

# Stemming
print("STEMMING:")
print("Porter:")
porterStemmer = PorterStemmer()
stem_count = 0
stems = []
for word in word_tokens:
    stem = porterStemmer.stem(word)
    if stem != word:
        stems.append(stem)
        if 0 < len(stems) <= 5:
            print(word + " => " + stem)

print("ect...")
print()

print("Lancaster:")
lancasterStemmer = LancasterStemmer()
stems = []
for word in word_tokens:
    stem = lancasterStemmer.stem(word)
    if stem != word:
        stems.append(stem)
        if 0 < len(stems) <= 5:
            print(word + " => " + stem)

print("ect...")
print()

print("Snowball:")
snowballStemmer = SnowballStemmer('english')
stems = []
for word in word_tokens:
    stem = snowballStemmer.stem(word)
    if stem != word:
        stems.append(stem)
        if 0 < len(stems) <= 5:
            print(word + " => " + stem)

print("ect...")
print()
# Lemmatization
print("LEMMATIZATION:")
lemmatizer = WordNetLemmatizer()
lems = []
for word in word_tokens:
    lem = lemmatizer.lemmatize(word)
    if lem != word:
        lems.append(lem)
        if 0 < len(lems) <= 5:
            print(word + " => " + lem)

print("ect...")
print()

# Trigrams
print("TRIGRAMS:")
trigrams = list(nltk.trigrams(word_tokens))
print(str(trigrams[:5]) + " ... and {} others.".format(len(trigrams[5:])))
print()

# NER
print("NAMED ENTITY RECOGNITION:")
named = []
for sentence in sentence_tokens:
    ner = ne_chunk(pos_tag(wordpunct_tokenize(sentence)))
    named.append(ner)
    if 0 < len(named) <= 3:
        print(ner)
print("ect...")
print()
