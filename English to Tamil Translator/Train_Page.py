import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
import io
import time
import unicodedata
import re
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
import pickle
import json

f = open('train.ta', encoding='utf8')
w1 = f.readlines()
print(len(w1))
print(w1[0:5])
g = open('train.en', encoding='utf8')
w2 = g.readlines()
print(len(w2))
print(w2[0:5])

NUM_SENTENCES = 25000

input_sentences = []
output_sentences = []

count = 0
for line in open('train.en', encoding='utf8'):
    count += 1

    if count > NUM_SENTENCES:
        break

    input_sentence = line.rstrip().strip("\n").strip('-') 
    input_sentences.append(input_sentence) 

count = 0

for line in open('train.ta', encoding='utf8'):
    count += 1

    if count > NUM_SENTENCES:
        break
    output_sentence =  line.rstrip().strip("\n").strip('-')
    
    from indicnlp.tokenize import indic_tokenize
    line = indic_tokenize.trivial_tokenize(output_sentence) 

    tokens = ['<sos>'] + line + ['<eos>']
    output_sentences.append(" ".join(tokens))

print(input_sentences[24999])
print(output_sentences[24999])
print(input_sentences[-1])
print(output_sentences[-1])

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

for i in range(len(input_sentences)):
   input_sentences[i] = preprocess_sentence(input_sentences[i])

print(input_sentences[3])
print(output_sentences[3])

def sample_function(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer.fit_on_texts(lang)
  tensor = lang_tokenizer.texts_to_sequences(lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor)
  return tensor, lang_tokenizer

def load_dataset(inp_lang, targ_lang):
  
  input_tensor, inp_lang_tokenizer = sample_function(inp_lang)
  target_tensor, targ_lang_tokenizer = sample_function(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(input_sentences, output_sentences)

max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

print(max_length_inp)
print(max_length_targ)
print(input_tensor[9])
print(input_sentences[9])
print(input_tensor[10])
print(input_sentences[10])
print(target_tensor[9])

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2,random_state=42)

print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))
      print ("%s ----> %d" % (lang.index_word[t], lang.word_index[lang.index_word[t]]))

print ("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print ()
print ("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[0])

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 256

vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) 

print(BUFFER_SIZE)
print(steps_per_epoch)
print(max_length_targ)
print(max_length_inp)
print(vocab_inp_size)
print(vocab_tar_size)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

def build_seq2seq(vocab_inp_size, vocab_tar_size, embedding_dim=256, units=256):

    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(vocab_inp_size, embedding_dim)(encoder_inputs)

    encoder_lstm = LSTM(units, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    dec_emb = Embedding(vocab_tar_size, embedding_dim)(decoder_inputs)

    decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    decoder_dense = Dense(vocab_tar_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model

model = build_seq2seq(
    vocab_inp_size=vocab_inp_size,
    vocab_tar_size=vocab_tar_size,
    embedding_dim=256,
    units=256
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

history = model.fit(
    [input_tensor_train, target_tensor_train[:, :-1]],  
    target_tensor_train[:, 1:],                         
    validation_data=(
        [input_tensor_val, target_tensor_val[:, :-1]],
        target_tensor_val[:, 1:]
    ),
    batch_size=64,
    epochs=10
)

model.save("eng_tam_translation_model.h5")

with open("inp_tokenizer.pkl", "wb") as f:
    pickle.dump(inp_lang, f)

with open("targ_tokenizer.pkl", "wb") as f:
    pickle.dump(targ_lang, f)

config = {
    "max_length_inp": int(max_length_inp),
    "max_length_targ": int(max_length_targ)
}

with open("config.json","w") as f:
    json.dump(config,f)