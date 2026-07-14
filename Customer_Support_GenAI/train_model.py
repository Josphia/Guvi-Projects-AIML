import pandas as pd
import numpy as np
import re
import nltk
import joblib
import contractions
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping

nltk.download('stopwords')
nltk.download('punkt')
nltk.download("wordnet")

print("Loading dataset...")
dataset = load_dataset("banking77")

train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()
df = pd.concat([train_df, test_df], ignore_index=True)

label_names = dataset["train"].features["label"].names

df["intent"] = df["label"].apply(lambda x: label_names[x])

def expand_contractions(text):
    return contractions.fix(text)

def clean_text(text):
    text = str(text).lower()
    text = text.replace('\\n', ' ').replace('\n', ' ')
    text = re.sub(r'<name>|<tel_num>|<email>|<.*?>', '', text) 
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

stop_words = set(stopwords.words('english'))
custom_stops = {'dear', 'hello', 'hi', 'team', 'support', 'please', 'thank', 'thanks', 'regards', 'sincerely'}
stop_words.update(custom_stops)

lemmatizer = WordNetLemmatizer()

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]
    return " ".join(filtered_tokens)

df = df.dropna(subset=['text'])
df['full_text'] = df['text']
df['full_text'] = df['full_text'].apply(expand_contractions)
df['full_text'] = df['full_text'].apply(clean_text)

df['text_for_ml'] = df['full_text'].apply(remove_stopwords) 
df['text_for_dl'] = df['full_text'] 

df = df[df['text_for_ml'].str.strip() != '']
df = df.drop_duplicates(subset=['text_for_ml'])
df = df.reset_index(drop=True)

le = LabelEncoder()
df['label'] = le.fit_transform(df['intent'])
joblib.dump(le, 'label_encoder.joblib') 

target_names = le.classes_
y = df['label'].values

X_train_idx, X_test_idx, y_train, y_test = train_test_split(
    df.index, y, test_size=0.2, stratify=y, random_state=42
)

X_train_ml = df.loc[X_train_idx, 'text_for_ml']
X_test_ml  = df.loc[X_test_idx, 'text_for_ml']
X_train_dl = df.loc[X_train_idx, 'text_for_dl']
X_test_dl  = df.loc[X_test_idx, 'text_for_dl']

print("Training Baseline Logistic Regression...")
tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9
)
X_train_tfidf = tfidf.fit_transform(X_train_ml)
X_test_tfidf = tfidf.transform(X_test_ml)
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
    
lr = LogisticRegression(
    C=2,
    solver='liblinear',
    max_iter=2000,
    class_weight='balanced'
)
lr.fit(X_train_tfidf, y_train)

cv_scores = cross_val_score(
    lr,
    X_train_tfidf,
    y_train,
    cv=5,
    scoring='accuracy'
)
print("\nCross Validation Accuracy:", cv_scores)
print("Average Cross Validation Accuracy: {:.2f}%".format(cv_scores.mean() * 100))

joblib.dump(lr, 'baseline_lr_model.joblib')

print("Tokenizing Text for LSTM...")
max_words = 10000
max_len = 250  
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_dl)
joblib.dump(tokenizer, 'tokenizer.joblib')

X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train_dl), maxlen=max_len)
X_test_padded = pad_sequences(tokenizer.texts_to_sequences(X_test_dl), maxlen=max_len)

model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0)), 
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(target_names), activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), 
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

raw_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
smoothed_weights = np.sqrt(raw_weights) 
class_weight_dict = dict(enumerate(smoothed_weights))

print("Training LSTM with Smoothed Class Weights...")
model.fit(
    X_train_padded, y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_test_padded, y_test),
    callbacks=[early_stop],
    class_weight=class_weight_dict
)

model.save("customer_model.h5")
print("\nSaved Model and Configurations Successfully!")

y_pred_lstm = np.argmax(model.predict(X_test_padded), axis=1)
y_pred_lr = lr.predict(X_test_tfidf)

print("\nConfusion Matrix (LSTM)")
print(confusion_matrix(y_test, y_pred_lstm))

print("\nConfusion Matrix (LR)")
print(confusion_matrix(y_test, y_pred_lr))

print("\nClassification Report (LSTM)")
print(classification_report(y_test, y_pred_lstm, target_names=target_names))

print("\nClassification Report (LR)")
print(classification_report(y_test, y_pred_lr, target_names=target_names))

print("\nCross Validation Accuracy:", cv_scores)