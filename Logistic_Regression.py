import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier


data = [
    {"text": "این محصول فوق‌العاده بود خیلی راضی هستم", "sentiment": "positive"},
    {"text": "کیفیت غذا عالی بود و پرسنل خیلی مهربان بودند", "sentiment": "positive"},
    {"text": "فیلم بسیار جذابی بود پیشنهاد می‌کنم ببینید", "sentiment": "positive"},
    {"text": "خدمات‌رسانی سریع و دقیق بود ممنون", "sentiment": "positive"},
    {"text": "از خریدم راضی هستم ارزش پولش رو داشت", "sentiment": "positive"},
    {
        "text": "برخورد فروشنده خیلی خوب بود و راهنمایی کامل داد",
        "sentiment": "positive",
    },
    {"text": "اپلیکیشن روان و بدون مشکل کار می‌کنه", "sentiment": "positive"},
    {"text": "هتل تمیز و آرام بود استراحت خوبی داشتیم", "sentiment": "positive"},
    {"text": "دوره آموزشی بسیار مفید و کاربردی بود", "sentiment": "positive"},
    {"text": "تحویل به موقع انجام شد و بسته‌بندی سالم بود", "sentiment": "positive"},
    {"text": "کیفیت صدا و تصویر فوق‌العاده بود", "sentiment": "positive"},
    {"text": "پشتیبانی سریع جواب داد و مشکل حل شد", "sentiment": "positive"},
    {"text": "قیمت مناسب و کیفیت بالا حتماً بازم خرید می‌کنم", "sentiment": "positive"},
    {"text": "غذا خوشمزه بود و محیط رستوران دلنشین", "sentiment": "positive"},
    {"text": "کتاب بسیار آموزنده و روان بود", "sentiment": "positive"},
    {"text": "ممنون از تیم خوبتون خیلی حرفه‌ای بودید", "sentiment": "positive"},
    {"text": "سفر خوبی بود همه چیز مرتب و منظم بود", "sentiment": "positive"},
    {"text": "این آموزش خیلی به من کمک کرد سپاسگزارم", "sentiment": "positive"},
    {"text": "محصول اصل و باکیفیت بود", "sentiment": "positive"},
    {"text": "تجربه خرید لذت‌بخشی داشتم پیشنهاد می‌کنم", "sentiment": "positive"},
    {"text": "دکتر خیلی دلسوز و ماهر بود", "sentiment": "positive"},
    {"text": "نرم‌افزار پرسرعت و پایدار است", "sentiment": "positive"},
    {"text": "طراحی سایت زیبا و کاربرپسند است", "sentiment": "positive"},
    {"text": "ارسال سریع و هزینه منطقی بود", "sentiment": "positive"},
    {"text": "با خانواده لحظات خوبی داشتیم عالی بود", "sentiment": "positive"},
    {"text": "کیفیت محصول بسیار پایین بود پشیمونم", "sentiment": "negative"},
    {"text": "غذا سرد و بی‌مزه بود اصلاً راضی نبودم", "sentiment": "negative"},
    {"text": "فیلم خسته‌کننده و بی‌معنی بود", "sentiment": "negative"},
    {"text": "خدمات بسیار ضعیف بود و کسی جواب نمی‌داد", "sentiment": "negative"},
    {"text": "از خریدم پشیمانم پول دور ریختن بود", "sentiment": "negative"},
    {"text": "برخورد پرسنل توهین‌آمیز و غیرحرفه‌ای بود", "sentiment": "negative"},
    {"text": "اپلیکیشن مدام کرش می‌کنه و کند است", "sentiment": "negative"},
    {"text": "اتاق هتل کثیف و پر سر و صدا بود", "sentiment": "negative"},
    {"text": "دوره آموزشی بی‌ارزش و تکراری بود", "sentiment": "negative"},
    {
        "text": "تحویل با تأخیر زیاد انجام شد و بسته آسیب دیده بود",
        "sentiment": "negative",
    },
    {"text": "کیفیت صدا افتضاح بود و تصویر قطع می‌شد", "sentiment": "negative"},
    {"text": "پشتیبانی هیچ جوابی نداد و مشکل باقی موند", "sentiment": "negative"},
    {"text": "قیمت بالا و کیفیت پایین ارزش خرید نداره", "sentiment": "negative"},
    {"text": "غذا بوی بدی می‌داد و محیط رستوران کثیف بود", "sentiment": "negative"},
    {"text": "کتاب پر از غلط و نامفهوم بود", "sentiment": "negative"},
    {"text": "تیم پشتیبانی بی‌مسئولیت و بی‌تجربه بودند", "sentiment": "negative"},
    {"text": "سفر افتضاحی بود همه چیز بی‌نظم بود", "sentiment": "negative"},
    {"text": "این آموزش هیچ کمکی نکرد وقت تلف کردن بود", "sentiment": "negative"},
    {"text": "محصول تقلبی و بی‌کیفیت بود", "sentiment": "negative"},
    {"text": "تجربه خرید بدی داشتم به هیچ‌کس پیشنهاد نمی‌کنم", "sentiment": "negative"},
    {"text": "دکتر بی‌توجه و عجول بود", "sentiment": "negative"},
    {"text": "نرم‌افزار کند و پر از باگ است", "sentiment": "negative"},
    {"text": "طراحی سایت زشت و گیج‌کننده است", "sentiment": "negative"},
    {"text": "ارسال خیلی دیر و هزینه بالا بود", "sentiment": "negative"},
    {"text": "لحظات بدی داشتیم اصلاً راضی نبودیم", "sentiment": "negative"},
]

df = pd.DataFrame(data)
df.sample(frac=1).reset_index(drop=True)  # mixing pos and neg sentimant

x = df["text"]
y = df["sentiment"]

countvec = CountVectorizer()
countvec_fit = countvec.fit_transform(x)
bag_of_words = pd.DataFrame(
    countvec_fit.toarray(), columns=countvec.get_feature_names_out()
)

x_train, x_test, y_train, y_test = train_test_split(
    bag_of_words, y, test_size=0.3, random_state=7
)

lr = LogisticRegression(random_state=1).fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

print(
    accuracy_score(y_pred_lr, y_test)
)  # 0.4 in my example => weak model but a good base

print(classification_report(y_test, y_pred_lr, zero_division=0))


# Trying to improve our accuracy using naive-bayes
nb = MultinomialNB().fit(x_train, y_train)
y_pred_nb = nb.predict(x_test)

print(accuracy_score(y_pred_nb, y_test))  # Stll 0.4 so didn`t work

# Trying to improve our accuracy with linear support vector machine (SVM)
svm = SGDClassifier().fit(x_train, y_train)
y_pred_svm = svm.predict(x_test)

print(accuracy_score(y_pred_svm, y_test))  # 0.333 This method didn't work either.
