from textblob import TextBlob
from textblob import Word
from textblob.wordnet import VERB, NOUN

from textblob.wordnet import Synset

import nltk
from nltk.corpus import wordnet as wn


# word = Word("pen")
# print(word.synsets)
# define = word.definitions
# print(define)
#
# octopus = Synset('octopus.n.02')
# shrimp = Synset('shrimp.n.03')
# simi = octopus.path_similarity(shrimp)
# print(simi)
#
# wiki = TextBlob("Python is a high-level, general-purpose programming language.")
# print(wiki.tags)
# print(wiki.noun_phrases)

print(wn.synsets('pen'))

print(wn.synset('penitentiary.n.01'))
print(wn.synset('penitentiary.n.01').definition())
print(wn.synset('dog.n.01').lemmas())
print(sorted(wn.langs()))
print(wn.synset('dog.n.01').lemma_names('ind'))

dog = wn.synset('dog.n.01')
print(dog.root_hypernyms())
print(wn.synset('dog.n.01').lowest_common_hypernyms(wn.synset('cat.n.01')))