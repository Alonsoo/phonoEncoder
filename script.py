import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense

df = pd.read_csv("manch_tok.csv")
df['phon_str'] = df['phon']
df['phon'] = df[['phon']].apply(lambda item : item[0].split("_"), axis = 1)

bnc = pd.read_csv("spok_bnc_typ.csv")
bnc = bnc.rename(columns={'phon' : 'phon_str'})

manch_typ = pd.read_csv("manch_typ.csv")
manch_typ = manch_typ.rename(columns={'phon' : 'phon_str'})

ids = ['anne', 'aran', 'becky', 'carl', 'domin', 'gail', 'joel', 'john', 'liz', 'nic', 'ruth', 'warr']
phonemes = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S","SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "<START>", "<END>", "<PAD>"]
vowels = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
num_phonemes = len(phonemes)

def encode_phon_one_hot(phon):
    vec = np.zeros(len(phonemes))
    vec[phonemes.index(phon)] = 1
    return vec

def decode_phon_one_hot(vec):
    index = np.argmax(vec)
    return phonemes[index]

def vec_to_word(seq):
    seq = seq[np.sum(seq, 1) == 1]
    return [decode_phon_one_hot(vec) for vec in seq]


words = np.unique(df['phon'].to_numpy())

max_word_length_input = max([len(word) for word in df['phon']])
max_word_length_target = max_word_length_input + 2

def get_dataset(id, stage):
    #Prepare Datasets
    input_words = df.copy()
    input_words = input_words[(input_words.id == id) & (input_words.stage == stage) & (input_words.set == 'MOT')][['phon']]

    target_words = input_words.copy()
    target_words['phon'] = target_words['phon'].transform(lambda word : ["<START>"] + word + ["<END>"])

    num_words = len(input_words)
    #max_word_length_input = max([len(word) for word in input_words['phon']])
    #max_word_length_target = max([len(word) for word in target_words['phon']])

    encoder_input = np.zeros((num_words, max_word_length_input, num_phonemes), dtype="float32")
    decoder_input = np.zeros((num_words, max_word_length_target, num_phonemes), dtype="float32")
    decoder_target = np.zeros((num_words, max_word_length_target, num_phonemes), dtype="float32")

    for i, (input_word, target_word) in enumerate(zip(input_words['phon'], target_words['phon'])):
        encoder_input[i] = [encode_phon_one_hot("<PAD>") for _ in range(max_word_length_input)]

        input_word = [encode_phon_one_hot(phon) for phon in input_word]
        encoder_input[i, :len(input_word)] = input_word
        
        #Reverse Inputs
        encoder_input[i] = np.flip(encoder_input[i])


        decoder_input[i] = [encode_phon_one_hot("<PAD>") for _ in range(max_word_length_target)]
        decoder_target[i] = [encode_phon_one_hot("<PAD>") for _ in range(max_word_length_target)]

        target_word = [encode_phon_one_hot(phon) for phon in target_word]
        decoder_input[i, :len(target_word)] = target_word
        decoder_target[i, :len(target_word) - 1] = target_word[1:]
    
    return encoder_input, decoder_input, decoder_target



def build_training_model(latent_dim):
    #latent_dim = 16

    encoder_inputs = Input(shape=(None, num_phonemes))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_phonemes))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the 
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_phonemes, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    return Model([encoder_inputs, decoder_inputs], decoder_outputs)



def build_inference_model(model, latent_dim):
    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = keras.Input(shape=(latent_dim,))
    decoder_state_input_c = keras.Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )
    
    return (encoder_model, decoder_model)



def train_model(model, epochs, e_input, d_input, d_target):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([e_input, d_input], d_target,
              batch_size=64,
              epochs=epochs,
              validation_split=0)


def predict_word(encoder_model, decoder_model, word):
    encoder_input = np.zeros((1, max_word_length_input, num_phonemes), dtype='float32')
    encoder_input[0] = [encode_phon_one_hot("<PAD>") for _ in range(max_word_length_input)]
    encoder_input[0,:len(word)] = [encode_phon_one_hot(phon) for phon in word]
    encoder_input[0] = np.flip(encoder_input[0])
    
    #print(word)
    #print(encoder_input)
    #print(vec_to_word(encoder_input[0]))
    
    states = encoder_model.predict(encoder_input, verbose = 0)
    
    predicted_phon = '<START>'
    predicted_word = list()
    
    
    while True:
        decoder_input = np.zeros((1, 1, num_phonemes), dtype='float32')
        decoder_input[0,0] = encode_phon_one_hot(predicted_phon)
        decoder_output, h, c = decoder_model.predict([decoder_input] + states, verbose = 0)
        
        states = [h, c]
    
        #print(decoder_output)
        predicted_phon = decode_phon_one_hot(decoder_output[0,0])
        #print(predicted_phon)
        
        if predicted_phon == "<END>" or len(predicted_word) >= max_word_length_input:
            break
            
        predicted_word.append(predicted_phon)
        
    return predicted_word 



def predict_words(encoder_model, decoder_model, words):
    encoder_input = np.full((len(words), max_word_length_input, num_phonemes), encode_phon_one_hot("<PAD>"), dtype='float32')
    for i, word in enumerate(words):
        encoder_input[i,:len(word)] = [encode_phon_one_hot(phon) for phon in word]
        encoder_input[i] = np.flip(encoder_input[i])
    
    states = encoder_model.predict(encoder_input, verbose = 0)
    
    predicted_phons = np.array(['<START>' for _ in range(len(words))])
    predicted_words = np.zeros((len(words), max_word_length_input), dtype='object')
    
    
    for i in range(max_word_length_input):
        decoder_input = np.zeros((len(words), 1, num_phonemes), dtype='float32')
        decoder_input = np.array([[encode_phon_one_hot(phon)] for phon in predicted_phons])

        decoder_output, h, c = decoder_model.predict([decoder_input] + states, verbose = 0)
        
        states = [h, c]
    

        predicted_phons = [decode_phon_one_hot(decoder_output[j,0]) for j in range(len(words))]  
        predicted_words[:,i] = predicted_phons
        
        
    return predicted_words


latent_dim = 64
models = [build_training_model(latent_dim) for id in ids]
inference_models = [build_inference_model(model, latent_dim) for model in models]

def evaluate_model(encoder, decoder):
    #encoder, decoder = model[0], model[1]
    words_learned = list()
    #for word in random.choices(words, k=100):
    for word in words:
        pred = predict_word(encoder, decoder, word)
        if pred == word:
            words_learned.append(word)
        #print(word)
        #print(pred)
    print("Words learned: {}".format(len(words_learned)))
    print("Percentage: {}".format(len(words_learned)/len(words)))
    return words_learned
    



def evaluate_model2(encoder, decoder):
    #encoder, decoder = model[0], model[1]
    words_learned = list()
    #test = random.choices(words, k=1000)
    predictions = predict_words(encoder, decoder, words)
    for i in range(len(words)):
        pred = predictions[i]
        pred = pred[(pred != '<END>')&(pred != '<PAD>')]
        if np.array_equal(words[i], pred):
            words_learned.append(pred)
    print("Words learned: {}".format(len(words_learned)))
    print("Percentage: {}".format(len(words_learned)/len(words)))
    return words_learned
    


model_df = pd.DataFrame(columns=('id', 'stage', 'phon', 'phon_str'))
for i in range(len(models)):
    already_learned = set()
    for j in range(20):
        stage = j+1
        encoder_input, decoder_input, decoder_target = get_dataset(ids[i], stage)
        train_model(models[i], 3, encoder_input, decoder_input, decoder_target)
        print('evaluating model {} at stage {}'.format(ids[i], stage))
        words_learned = evaluate_model2(inference_models[i][0], inference_models[i][1])
        for word in words_learned:
            word_str = '_'.join(word)
            if not (word_str in already_learned):
                model_df.loc[len(model_df.index)] = [ids[i], stage, word, word_str]
                already_learned.add(word_str)
        del encoder_input
        del decoder_input
        del decoder_target
        del words_learned
        model_df.to_csv('model_df.csv', index=False)

model_df.to_csv('model_df.csv', index=False)