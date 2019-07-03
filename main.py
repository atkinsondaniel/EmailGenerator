#%%
import tensorflow as tf
import numpy as np
import datetime
import os
#%%
DATA_HOME = "C:/Users/592123/Documents/DS Projects/EmailGenLSTM/data/"

with open(DATA_HOME + 'clean_data.txt', 'r') as f:
    text = f.read().lower()
print('corpus length:', len(text))
#%%
words = [w for w in text.split(' ') if w.strip() !='' or w == '\n']
print("Text is {} words long".format(len(words)))
print(text[10000:10100])
#%%
vocab = sorted(set(text))
print ('There are {} unique characters'.format(len(vocab)))
char2int = {c:i for i, c in enumerate(vocab)}
int2char = np.array(vocab)
print('Vector:\n')
for char,_ in zip(char2int, range(len(vocab))):
    print(' {:4s}: {:3d},'.format(repr(char), char2int[char]))
#%%
text_as_int = np.array([char2int[ch] for ch in text], dtype=np.int32)
print ('{}\n mapped to integers:\n {}'.format(repr(text[:100]), text_as_int[:100]))
#%%
tr_text = text_as_int[:12480000] 
val_text = text_as_int[12480000:] 
print(text_as_int.shape, tr_text.shape, val_text.shape)
#%%
batch_size = 64
buffer_size = 10000
embedding_dim = 256
epochs = 50
seq_length = 200
examples_per_epoch = len(text)//seq_length
#lr = 0.001 #will use default for Adam optimizer
rnn_units = 1024
vocab_size = len(vocab)

tr_char_dataset = tf.data.Dataset.from_tensor_slices(tr_text)
val_char_dataset = tf.data.Dataset.from_tensor_slices(val_text)
tr_sequences = tr_char_dataset.batch(seq_length+1, drop_remainder=True)
val_sequences = val_char_dataset.batch(seq_length+1, drop_remainder=True)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
tr_dataset = tr_sequences.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True)
val_dataset = val_sequences.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True)
print(tr_dataset, val_dataset)
#%%
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
     
    model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(rnn_units,
                                 return_sequences=True,
                                 stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(rnn_units,
                                 return_sequences=True,
                                 stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(vocab_size)
            ])
    return model
model = build_model(
        vocab_size = len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=batch_size)
#%%
model.summary()
for input_example_batch, target_example_batch in tr_dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape)
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Loss:      ", example_batch_loss.numpy().mean())
#%%
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=loss)
patience = 10
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
#%%
checkpoint_dir = DATA_HOME + 'checkpoints'+ datetime.datetime.now().strftime("_%Y.%m.%d-%H.%M.%S/")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
history = model.fit(tr_dataset, epochs=epochs, callbacks=[checkpoint_callback, early_stop] , validation_data=val_dataset)
print ("Training stopped as there was no improvement after {} epochs".format(patience))
