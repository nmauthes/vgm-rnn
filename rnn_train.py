'''

Train the RNN on a collection of MIDI files.

:authors Nicolas Mauthes

'''

import os
import time
import argparse
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from midi_parser import MIDIError, filter_midi_files, pretty_midi_to_piano_roll, stringify


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
MAX_EPOCHS = 200

LSTM_UNITS = 256

SAVE_CHECKPOINTS = False
SAVE_GRAPH = False

SAVED_DICT_PATH = 'chord_dict.pkl'

# |-------------------------------------|


def build_model(vocab_size): # TODO expand architecture
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=SEQUENCE_LENGTH))
    model.add(LSTM(LSTM_UNITS, return_sequences=True))
    model.add(LSTM(LSTM_UNITS))
    model.add(Dense(LSTM_UNITS, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))

    return model


def split_xy(data, seq_length):
    x = []
    y = []

    # Pad array with zeros so we get consistent sequence lengths
    # data = list(data + (np.zeros_like((data[0])) * seq_length)) # TODO Fix?

    # Split data into training/labels
    for i in range(0, len(data) - seq_length, seq_length): # TODO change step to measures?
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
        # y.append(data[i + 1: i + seq_length + 1])

    x = np.asarray(x)
    y = np.asarray(y)

    return x, y


def get_formatted_time(time_in_sec):
    hours = int(time_in_sec // 3600)
    minutes = int(time_in_sec % 3600 // 60)
    seconds = int(time_in_sec % 60)

    return f'{"{0:0>2}".format(hours)}h:{"{0:0>2}".format(minutes)}m:{"{0:0>2}".format(seconds)}s'


# |---------- COMMAND LINE ARGS ----------|

parser = argparse.ArgumentParser(
    description='Build and train a new RNN model',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='See the README for further questions'
)

parser.add_argument(
    '--max_epochs',
    default=MAX_EPOCHS,
    type=int,
    help='The maximum number of epochs to allow for training.'
)
parser.add_argument(
    '--save_checkpoints',
    default=SAVE_CHECKPOINTS,
    help='Whether to save model checkpoints during training.'
)
parser.add_argument(
    '--save_graph',
    default=SAVE_GRAPH,
    help='Whether to save a Tensorboard graph of the model.'
)

# |---------------------------------------|


if __name__ == '__main__':
    args = parser.parse_args()

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

    nonzero_count = np.sum(np.any(midi_data, axis=1)) # For determining average polyphony
    print(f'Average polyphony: {round(np.sum(midi_data) / nonzero_count, 2)} notes per chord')

    print('-' * 25)
    print('Preparing data for training...')

    # Clamp MIDI note range and convert to string representation
    midi_data = midi_data[:, MIN_MIDI_NOTE:MAX_MIDI_NOTE + 1]
    midi_data = stringify(midi_data)

    # Get unique tokens and create dict mapping each to an integer
    unique_tokens = np.unique(midi_data, axis=0)
    vocab_size = len(unique_tokens) # TODO Add 1 to size?

    chord_dict = dict((chord, i) for i, chord in enumerate(unique_tokens))

    if not os.path.exists(SAVED_DICT_PATH):
        with open(SAVED_DICT_PATH, 'wb') as f:
            pickle.dump(chord_dict, f)

    print(f'Number of unique tokens: {vocab_size}')

    # Finally, create integer sequences for input to embedding layer
    midi_data = np.asarray([chord_dict[chord] for chord in midi_data])

    training_data, label_data = split_xy(midi_data, SEQUENCE_LENGTH)
    label_data = to_categorical(label_data, num_classes=vocab_size)

    print(f'Number of sequences: {len(training_data)} ({SEQUENCE_LENGTH} timesteps per sequence)')
    print('-' * 25)

    # Build and compile model
    model = build_model(vocab_size)
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER)

    # Callbacks and regularization
    callbacks = []

    if args.save_checkpoints:
        checkpoints = ModelCheckpoint(MODEL_FOLDER, monitor='loss', save_best_only=True, save_weights_only=True)
        callbacks.append(checkpoints)

    if args.save_graph:
        tb = TensorBoard(log_dir=MODEL_FOLDER)
        callbacks.append(tb)

    early_stopping = EarlyStopping(monitor='loss', min_delta=0.01, patience=10)
    callbacks.append(early_stopping)

    # Train model, then save weights
    start_time = time.time()
    model.fit(training_data, label_data, batch_size=BATCH_SIZE, epochs=args.max_epochs, callbacks=callbacks) # TODO shuffle?
    training_time = time.time() - start_time

    #model.save_weights(os.path.join(MODEL_FOLDER, SAVED_WEIGHTS_PATH)) # TODO validation?

    print()
    print(f'Training time was {get_formatted_time(training_time)}')
    print(f'Weights saved as \'{SAVED_WEIGHTS_PATH}\'')
