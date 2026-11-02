from hazm import *
import gensim
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from gensim.models import LsiModel
import pandas as pd
import re
import matplotlib as plt

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

# biulding LSA model

lsa_model = LsiModel(doc_term, num_topics=num_topic, id2word=dictionary)
lsa_model.print_topics(num_topics=num_topic, num_words=2)

# compute coherence score

coherence_val = []
model_list = []

min_topics = 2
max_topics = 5

for num_topics_i in range(min_topics, max_topics + 1):
    model = LsiModel(
        doc_term, num_topics=num_topics_i, id2word=dictionary, random_seed=0
    )
    model_list.append(model)
    coherence_model = CoherenceModel(
        model=model, texts=article, dictionary=dictionary, coherence="c_v"
    )
    coherence_val.append(coherence_model.get_coherence())

# ploting the best topic number

plt.plot(range(min_topics, max_topics + 1), coherence_val)
plt.xlabel("Number Of Topics")
plt.ylabel("Coherence Score")
plt.legend(("Coherence_values"), loc="best")

# final model with best topic number for humans to undrestand

final_n_topic = 2  # for example
lsamodel_f = LsiModel(doc_term, num_topics=final_n_topic, id2word=dictionary)
lsamodel_f.print_topics(num_topics=final_n_topic, num_words=2)
