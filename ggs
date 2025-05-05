from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from nltk.stem.snowball import SnowballStemmer
import re

data = pd.read_csv("arxiv_data/arxiv_dataset.csv", index_col=0)

X_train = data.loc[:2999, 'sampled_sentence']
X_test = data.loc[3000:, 'sampled_sentence']
y_train = data.loc[:2999, 'paper_section']

# Инициализируем стеммер
stemmer = SnowballStemmer("english")

# Кастомная токенизация + стемминг
def tokenize_and_stem(text):
    # Базовая нормализация текста
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.strip().split()
    return [stemmer.stem(token) for token in tokens if len(token) > 2]

vectorizer = TfidfVectorizer(
    tokenizer=tokenize_and_stem,
    ngram_range=(1, 2),          # unigrams + bigrams
    stop_words='english',        # убираем частые слова
    max_df=0.06,                 # убираем супервстречающиеся слова
)

X_train_vec = vectorizer.fit_transform(X_train)  # sparse matrix
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
