import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import json
import re
import unicodedata

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


model = load_model("eng_tam_translation_model.h5")

with open("inp_tokenizer.pkl","rb") as f:
    inp_lang = pickle.load(f)

with open("targ_tokenizer.pkl","rb") as f:
    targ_lang = pickle.load(f)

with open("config.json") as f:
    config = json.load(f)

max_length_inp = config["max_length_inp"]
max_length_targ = config["max_length_targ"]


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    w = '<sos> ' + w + ' <eos>'
    return w


reverse_target_index = {i: w for w, i in targ_lang.word_index.items()}


def translate(sentence):

    sentence = preprocess_sentence(sentence)

    seq = inp_lang.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_length_inp, padding="post")

    decoder_input = [targ_lang.word_index["<sos>"]]

    output_sentence = []

    for _ in range(max_length_targ):

        decoder_seq = pad_sequences([decoder_input],
                                    maxlen=max_length_targ,
                                    padding="post")

        preds = model.predict([seq, decoder_seq], verbose=0)

        pred_id = np.argmax(preds[0, len(decoder_input)-1, :])

        word = reverse_target_index.get(pred_id,"")

        if word == "<eos>" or word == "":
            break

        output_sentence.append(word)

        decoder_input.append(pred_id)

    return " ".join(output_sentence)


st.title("English → Tamil Translator")

text = st.text_input("Enter English sentence")

if st.button("Translate"):

    if text.strip() != "":
        translation = translate(text)
        st.success(translation)