'''

Train the RNN on a collection of MIDI files.

'''

import os
import argparse

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop

from midi_parser import MIDIError, filter_midi_files, pretty_midi_to_numpy_array


DATA_FOLDERS = ['nes_data']
TRAINING_DATA_PATH = 'training_data.npy'

ALLOWED_TIME_SIGS = ['4/4']

# Note that pretty_midi assigns each of the major and minor keys a number from 0 to 23
# starting with all 12 major keys followed by all 12 minor keys such that C maj = 0, Db maj = 1 ... C min = 11, etc.

ALLOWED_KEYS = ['Db Major', 'D Major', 'Eb Major', 'E Major', 'F Major', 'Gb Major', 'G Major',
                'Ab Major', 'A Major', 'Bb Major', 'B Major', 'C minor', 'C# minor', 'D minor', 'Eb minor',
                'E minor', 'F minor', 'F# minor', 'G minor', 'G# minor', 'A minor', 'Bb minor', 'B minor']

SUBDIVISION = 8
MAX_DURATION = SUBDIVISION * 4 # Corresponds to 1 whole note


if __name__ == '__main__':
    if os.path.exists(TRAINING_DATA_PATH):
        print('Loading training data...')
        training_data = np.load(TRAINING_DATA_PATH) # Load saved training data if available
    else:
        print('Building training set. This might take a while...')
        training_data = np.empty(shape=(128, 0, 2)) # Otherwise gather MIDIs and build training set

        errors = 0
        for i, folder in enumerate(DATA_FOLDERS):
            print(f'Processing folder \'{folder}\' ({i + 1}/{len(DATA_FOLDERS)})')

            if os.path.exists(folder):
                midis = filter_midi_files(folder, ALLOWED_TIME_SIGS, ALLOWED_KEYS)

                for mid in midis:
                    try:
                        arr = pretty_midi_to_numpy_array(mid, subdivision=SUBDIVISION, transpose_notes=True)
                        training_data = np.concatenate((training_data,arr), axis=1)
                    except MIDIError:
                        errors += 1
            else:
                raise Exception('Data folder not found!')

            np.save(TRAINING_DATA_PATH, training_data) # Serialize array containing training data for future use

    model = Sequential()
    #model.add(LSTM(128, input_shape=(training_data.shape[1], 2))



