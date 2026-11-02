from hazm import *
import gensim
import gensim.corpora as corpora
import pandas as pd
import re

texts = [
    "پایتون یک زبان برنامه نویسی محبوب است.",
    "یادگیری ماشین و هوش مصنوعی بسیار جذاب هستند.",
    "شبکه های عصبی در یادگیری عمیق استفاده می شوند.",
    "فوتبال یکی از محبوب ترین ورزش های جهان است.",
    "تیم فوتبال ما در مسابقه پیروز شد.",
    "یادگیری پایتون برای تحلیل داده مفید است.",
    "هوش مصنوعی در تحلیل داده کاربرد دارد.",
    "بازیکنان فوتبال برای مسابقه تمرین می کنند.",
]

df = pd.DataFrame(texts, columns=["news"])

# remove punc and lower if needed (FA doesn`t need lower() but I added just to be there`)
df["article"] = df["news"].apply(lambda x: re.sub(r"[^\w\s]", "", x.lower()))
article = df["article"]

# normalize
normalizer = Normalizer()
article = article.apply(lambda x: normalizer.normalize(x))

# remove stop words
stop_words = set(stopwords_list())
article = article.apply(
    lambda x: " ".join([word for word in x.split() if word not in stop_words])
)

# tokenize
article = article.apply(lambda x: word_tokenize(x))

# stemming will not work here and mess our data and lemmatizing works better

lemmatizer = Lemmatizer()
article = article.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# biulding LDA model
dictionary = corpora.Dictionary(article)
doc_term = [dictionary.doc2bow(text) for text in article]

num_topic = 2
lda_model = gensim.models.LdaModel(
    corpus=doc_term, id2word=dictionary, num_topics=num_topic
)

lda_model.print_topics(num_topics=num_topic, num_words=2)
