from hazm import *

# normalizing

Sentence = "من دوست دارم در شرکتها کار کنم."
normalizer = Normalizer()
normalized_text = normalizer.normalize(Sentence)

# stemming
s1 = Sentence.split()
stemmer = Stemmer()
stem_word = stemmer.stem(s1[4])

# lemmatizing

lemmatizer = Lemmatizer()
lem_word = lemmatizer.lemmatize(s1[1])

#
