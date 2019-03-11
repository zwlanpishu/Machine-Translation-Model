from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam

'''a frequently used function'''
from keras.utils import to_categorical
from keras.models import load_model, Model
from faker import Faker
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import random

''' number of samples '''
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

''' preprocess the dataset, padding the X and Y '''
Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)

''' machine translation with attention '''

''' define the shared layers '''
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis = -1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)

''' define one step attention '''
def one_step_attention(a, s_prev) : 
    
    ''' Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a"  '''
    s_prev = repeator(s_prev)

    ''' after the concatenation, the shape of tensor is (m, Tx, n_s + 2 * n_a) '''
    concat = concatenator([a, s_prev])

    ''' put the concatenation tensor into the small fully connected network ''' 
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas, a])

    return context

''' define the layers needed to construct model '''
n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation = softmax)

''' define the specific model '''
def model(Tx, Ty, n_a, n_s, human_vocab, machine_vocab) : 
    X = Input(shape = (Tx, len(human_vocab)))
    s0 = Input(shape = (n_s, ), name = "s0")
    c0 = Input(shape = (n_s, ), name = "c0")
    s_prev = s0
    c_prev = c0

    ''' initialize the outputs '''
    outputs = []

    ''' define pre-attention bi-LSTM '''
    a = Bidirectional(LSTM(n_a, return_sequences = True))(X)

    for t in range(Ty) : 
        context = one_step_attention(a, s_prev)
        s_prev, _ , c_prev = post_activation_LSTM_cell(context, initial_state = [s_prev, c_prev])
        out = output_layer(s_prev)
        outputs.append(out)

    model = Model(inputs = [X, s0, c0], outputs = outputs)
    return model

mt_model = model(Tx, Ty, n_a, n_s, human_vocab, machine_vocab)
mt_model.summary()
mt_model.compile(loss = "categorical_crossentropy", 
                 optimizer = Adam(lr = 0.005, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01),
                 metrics = ["accuracy"])



s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))
print(outputs)

''' train the model '''
#mt_model.fit([Xoh, s0, c0], outputs, epochs = 30, batch_size = 100)


''' for saving time just load a pre-trained model '''
mt_model.load_weights("models/model.h5")

EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
    source = source.reshape((1, 30, 37))
    prediction = mt_model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    
    print("source:", example)
    print("output:", ''.join(output))

attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday April 08 1993", num = 6, n_s = 128)