from textblob import TextBlob
from textblob import Word
from textblob.wordnet import VERB, NOUN






word = Word("death")
print(word.synsets)
define = word.definitions
print(define)

wiki = TextBlob("Python is a high-level, general-purpose programming language.")
print(wiki.tags)
print(wiki.noun_phrases)

