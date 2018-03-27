'''

Train the RNN on a collection of MIDI files.

:authors Nicolas Mauthes

'''

import os
import argparse
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from midi_parser import MIDIError, filter_midi_files, pretty_midi_to_piano_roll


# |---------- MIDI PARAMS ----------|

DATA_FOLDERS = ['nes_data']
MIDI_DATA_PATH = 'training_data.npy'

ALLOWED_TIME_SIGS = ['4/4']
ALLOWED_KEYS = ['Db Major', 'D Major', 'Eb Major', 'E Major', 'F Major', 'Gb Major', 'G Major',
                'Ab Major', 'A Major', 'Bb Major', 'B Major', 'C minor', 'C# minor', 'D minor', 'Eb minor',
                'E minor', 'F minor', 'F# minor', 'G minor', 'G# minor', 'A minor', 'Bb minor', 'B minor']

MIN_MIDI_NOTE = 36 # C2
MAX_MIDI_NOTE = 84 # C6
MIDI_NOTE_RANGE = MAX_MIDI_NOTE - MIN_MIDI_NOTE + 1

SUBDIVISION = 4 # Number of steps per quarter note (e.g. 4 = 16th notes)
NOTES_IN_MEASURE = SUBDIVISION * 4
MAX_DURATION = NOTES_IN_MEASURE # Corresponds to 1 whole note


# |---------- TRAINING PARAMS ----------|

MODEL_FOLDER = 'model'
SAVED_WEIGHTS_PATH = 'rnn_weights.h5'

SEQUENCE_LENGTH = SUBDIVISION * 4 * 4

LOSS_FUNCTION = 'categorical_crossentropy'
LEARNING_RATE = 0.01
OPTIMIZER = Adam(lr=LEARNING_RATE)

BATCH_SIZE = 50
MAX_EPOCHS = 5

LSTM_UNITS = 256
DECODER_LAYERS = 2

SAVE_CHECKPOINTS = False
EARLY_STOPPING = False

# |-------------------------------------|


def build_model(): # TODO expand architecture
    model = Sequential()
    model.add(LSTM(LSTM_UNITS, input_shape=(SEQUENCE_LENGTH, MIDI_NOTE_RANGE), activation='tanh', return_sequences=True))
    model.add(Dense(MIDI_NOTE_RANGE))
    model.add(Activation('softmax'))

    return model


def split_xy(data, seq_length):
    x = []
    y = []

    # Pad array with zeros so we get consistent sequence lengths
    data = list(data) + (np.zeros_like((data[0])) * seq_length)

    # Split data into training/labels
    for i in range(0, len(data) - seq_length, seq_length): # TODO change step to measures?
        x.append(data[i:i + seq_length])
        y.append(data[i + 1: i + seq_length + 1])

    x = np.asarray(x)
    y = np.asarray(y)

    return x, y


if __name__ == '__main__':
    # Load/build midi data
    if os.path.exists(MIDI_DATA_PATH):
        print('Loading data...')
        midi_data = np.load(MIDI_DATA_PATH) # Load saved training data if available
    else:
        print('Building training set. This might take a while...')
        midi_data = np.empty(shape=(0, 128)) # Otherwise gather MIDIs and build training set

        errors = 0
        for i, folder in enumerate(DATA_FOLDERS):
            print(f'Processing folder \'{folder}\' ({i + 1}/{len(DATA_FOLDERS)})')

            if os.path.exists(folder):
                midis = filter_midi_files(folder, ALLOWED_TIME_SIGS, ALLOWED_KEYS)

                for mid in midis:
                    try:
                        arr = pretty_midi_to_piano_roll(mid, subdivision=SUBDIVISION, transpose_notes=True)
                        midi_data = np.concatenate((midi_data, arr), axis=0)
                    except MIDIError:
                        errors += 1
            else:
                raise Exception('Data folder not found!')

            if errors:
                print(f'{errors} errors occurred')

            print('Saving data...')
            np.save(MIDI_DATA_PATH, midi_data) # Serialize array containing training data for future use

    print(f'Total timesteps: {len(midi_data)}')

    print('-' * 25)
    print('Preparing data for training...')

    training_data, label_data = split_xy(midi_data, SEQUENCE_LENGTH)

    # Clamp MIDI note range
    training_data = training_data[:, :, MIN_MIDI_NOTE:MAX_MIDI_NOTE + 1]
    label_data = label_data[:, :, MIN_MIDI_NOTE:MAX_MIDI_NOTE + 1]

    print(f'Number of sequences: {len(training_data)} ({SEQUENCE_LENGTH} timesteps per sequence)')
    print('-' * 25)

    # Build and compile model
    model = build_model()
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER)

    # TODO early stopping, checkpoints

    # Callbacks and regularization
    callbacks = []

    if EARLY_STOPPING:
        early_stopping = EarlyStopping(monitor='loss', min_delta=0.01, patience=10)
        callbacks.append(early_stopping)

    if SAVE_CHECKPOINTS:
        checkpoints = ModelCheckpoint(MODEL_FOLDER, monitor='loss', save_best_only=True, save_weights_only=True)
        callbacks.append(checkpoints)

    # Train model, then save weights
    model.fit(training_data, label_data, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, callbacks=callbacks) # TODO shuffle?
    model.save_weights(os.path.join(MODEL_FOLDER, SAVED_WEIGHTS_PATH)) # TODO validation?

    print()
    print(f'Weights saved as \'{SAVED_WEIGHTS_PATH}\'')
