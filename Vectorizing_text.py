import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# bag of words method
texts = [
    "سلام!!! حال شما چطوره؟؟؟",
    "من    دارم    پایتون    یاد می‌گیرم.",
    "امروز هوا خیلی خوبه 😊🌞",
    "قیمت این محصول 250000 تومان است!!!",
    "شماره من 09123456789 است.",
    "آدرس سایت من https://example.com است.",
    "ایمیل من test123@example.com است.",
    "یادگیری NLP خیلی جذابه!!! #Python #NLP",
    "این محصول عالیه 😍 ولی قیمتش گرونه!!!",
    "پایتون\t\tیک زبان برنامه‌نویسی است.\nخیلی کاربردی است.",
    "من كتاب فارسي مي‌خوانم.",
    "این متن... شامل؟؟؟ علائم!!! مختلفـــــ است.",
    "نسخه Python 3.12.10 و نسخه NumPy 1.26.4",
    "سلام @user123، به کانال ما خوش آمدی!",
    "متن با پرانتز (آزمایشی) و [براکت] و {آکولاد}",
]

CountVect = CountVectorizer()
CountVect_fit = CountVect.fit_transform(texts)
bag_of_words = pd.DataFrame(
    CountVect_fit.toarray(), columns=CountVect.get_feature_names_out()
)

# tf-idf method
tfidfvec = TfidfVectorizer()
tfidfvec_fit = tfidfvec.fit_transform(texts)
tfidf_bag = pd.DataFrame(
    tfidfvec_fit.toarray(), columns=tfidfvec.get_feature_names_out()
)
