import pandas as pd
import numpy as np
import re
import nltk
import joblib
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

nltk.download('stopwords')
nltk.download('punkt')

dataset = load_dataset("Tobi-Bueck/customer-support-tickets")
df = dataset['train'].to_pandas()

contractions_dict = {"don't": "do not", "can't": "cannot", "i'm": "i am", "it's": "it is", "i've": "i have", "you're": "you are", "doesn't": "does not"}

def expand_contractions(text):
    for word, expanded in contractions_dict.items():
        text = re.sub(r'\b' + word + r'\b', expanded, text)
    return text

stop_words = set(stopwords.words('english'))
custom_stops = {'dear', 'hello', 'hi', 'team', 'support', 'please', 'thank', 'thanks', 'regards', 'sincerely', 'writing', 'message'}
stop_words.update(custom_stops)

def clean_text(text):
    text = str(text).lower()
    text = text.replace('\\n', ' ').replace('\n', ' ')
    text = re.sub(r'<name>', '[NAME]', text)
    text = re.sub(r'<tel_num>', '[PHONE]', text)
    text = re.sub(r'<email>', '[EMAIL]', text)
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

df = df[df['language'] == 'en'].copy()
df = df.dropna(subset=['body', 'queue'])
df['full_text'] = df['subject'].fillna('') + " " + df['body'].fillna('')
df['full_text'] = df['full_text'].apply(expand_contractions)
df['full_text'] = df['full_text'].apply(clean_text)
df['text_for_ml'] = df['full_text'].apply(remove_stopwords) 
df['text_for_dl'] = df['full_text'] 
df = df[df['text_for_ml'].str.strip() != '']
df = df.drop_duplicates(subset=['text_for_ml'])
df = df.reset_index(drop=True)

le = LabelEncoder()
df['label'] = le.fit_transform(df['queue'])
joblib.dump(le, 'label_encoder.joblib') 

target_names = le.classes_
y = df['label'].values

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['text_for_dl'], y, test_size=0.2, stratify=y, random_state=42
)

print("Creating TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')

print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)
joblib.dump(lr, 'baseline_lr_model.joblib')
y_pred_lr = lr.predict(X_test_tfidf)

print("Creating Tokenizer...")
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)
joblib.dump(tokenizer, 'tokenizer.joblib')

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=max_len)
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=max_len)

# model = Sequential([
#     Embedding(max_words, 128, input_length=max_len),
#     SpatialDropout1D(0.2),
#     Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)),
#     Dense(len(target_names), activation='softmax')
# ])

model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)),
    Dense(64, activation='relu'),
    Dense(len(target_names), activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y_train),
#     y=y_train
# )

# class_weights = dict(enumerate(class_weights))




early_stop = EarlyStopping(patience=3, restore_best_weights=True)

print("Training LSTM...")
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    #class_weight=class_weights
)

model.save("customer_model.keras")
print("Saved successfully!")