from hazm import *
import re

# normalizing
Sentence = "من دوست دارم در شرکتها کار کنم."
normalizer = Normalizer()
normalized_text = normalizer.normalize(Sentence)

# stemming
s1 = Sentence.split()
stemmer = Stemmer()
stem_word = stemmer.stem(s1[1])

# lemmatizing
lemmatizer = Lemmatizer()
lem_word = lemmatizer.lemmatize(s1[1])

# Sentence tokenizing
sentence_tokenizer = SentenceTokenizer()
sent_tokenize = sentence_tokenizer.tokenize(Sentence)

# Word tokenizing
word_tokenizer = WordTokenizer()
word_token = word_tokenizer.tokenize(Sentence)

# remove punctuation


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

# Regex control
sites = []

pattern = r"https?://[^\s]+"  # almost detect every normal website

for text in texts:
    found_sites = re.findall(pattern, text)
    sites.extend(found_sites)

print(sites)  # ['https://example.com']

# delete every text that start with non-character element
no_punc_text = []
pattern2 = r"[^\w\s]"

for text in texts:
    no_punc_str = re.sub(pattern2, "", text)
    no_punc_text.append(no_punc_str)

print(no_punc_text)

# findind numbers both in FA or EN
nums = []
for text in texts:
    numbers = re.findall(r"[۰-۹0-9]+", text)
    if numbers != []:
        nums.append(numbers)

print(nums)

# striping extra spaces
for text in texts:
    cleaned = re.sub(r"\s+", " ", text).strip()
    print(cleaned)
    print("-" * 50)
