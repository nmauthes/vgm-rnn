'''

Train the RNN on a collection of MIDI files.

:authors Nicolas Mauthes

'''

import os
import time
import argparse

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

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
MAX_EPOCHS = 200

LSTM_UNITS = 256
DECODER_LAYERS = 2

SAVE_CHECKPOINTS = False
SAVE_GRAPH = False

# |-------------------------------------|


def build_model():
    ''' Build the RNN model to be used during training and generation. '''

    model = Sequential()
    model.add(LSTM(LSTM_UNITS, input_shape=(SEQUENCE_LENGTH, MIDI_NOTE_RANGE), activation='tanh', return_sequences=True))
    model.add(Dense(MIDI_NOTE_RANGE))
    model.add(Activation('softmax'))

    return model


def split_xy(data, seq_length):
    ''' Split the dataset into a training set and corresponding set of labels. '''

    x = []
    y = []

    # Pad data with zeros to get consistent sequence lengths
    zero_pad = np.zeros((SEQUENCE_LENGTH - data.shape[0] % SEQUENCE_LENGTH, data.shape[1]))
    data = np.append(data, zero_pad, axis=0)

    # Split data into training/labels
    for i in range(0, len(data) - seq_length, seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length: i + seq_length * 2])

    x = np.asarray(x)
    y = np.asarray(y)

    return x, y


def get_formatted_time(time_in_sec):
    ''' Utility function for displaying time durations nicely. '''

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
        print('Loading data...') # Load saved dataset if available
        midi_data = np.load(MIDI_DATA_PATH)
    else:
        print('Building training set. This might take a while...') # Otherwise gather MIDIs and build dataset
        midi_data = np.empty(shape=(0, 128))

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
            np.save(MIDI_DATA_PATH, midi_data) # Serialize array containing dataset for future use

    print(f'Total timesteps: {len(midi_data)}')

    nonzero_count = np.sum(np.any(midi_data, axis=1))
    print(f'Average polyphony: {round(np.sum(midi_data) / nonzero_count, 2)} notes per chord')

    print('-' * 25)
    print('Preparing data for training...')

    # Split data into training set/labels
    training_data, label_data = split_xy(midi_data, SEQUENCE_LENGTH)

    # Clamp MIDI note range
    training_data = training_data[:, :, MIN_MIDI_NOTE:MAX_MIDI_NOTE + 1]
    label_data = label_data[:, :, MIN_MIDI_NOTE:MAX_MIDI_NOTE + 1]

    print(f'Number of sequences: {len(training_data)} ({SEQUENCE_LENGTH} timesteps per sequence)')
    print('-' * 25)

    # Build and compile model
    model = build_model()
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER)

    # Configure callbacks and regularization
    callbacks = []

    if args.save_checkpoints:
        checkpoints = ModelCheckpoint(MODEL_FOLDER, monitor='loss', save_best_only=True, save_weights_only=True)
        callbacks.append(checkpoints)

    if args.save_graph:
        tb = TensorBoard(log_dir=MODEL_FOLDER)
        callbacks.append(tb)

    early_stopping = EarlyStopping(monitor='loss', min_delta=0.01, patience=10)
    callbacks.append(early_stopping)

    # Train model, then save weights when done
    start_time = time.time()
    model.fit(training_data, label_data, batch_size=BATCH_SIZE, epochs=args.max_epochs, callbacks=callbacks)
    training_time = time.time() - start_time

    model.save_weights(os.path.join(MODEL_FOLDER, SAVED_WEIGHTS_PATH))

    print()
    print(f'Training time was {get_formatted_time(training_time)}')
    print(f'Weights saved as \'{SAVED_WEIGHTS_PATH}\'')
