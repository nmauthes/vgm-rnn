'''

Train the RNN on a collection of MIDI files.

'''

import os
import pickle

import numpy as np
import pretty_midi

from midi_parser import MIDIError, filter_midi_files, pretty_midi_to_numpy_array


ALLOWED_TIME_SIGS = ['4/4']

# Note that pretty_midi assigns each of the major and minor keys a number from 0 to 23
# starting with all 12 major keys followed by all 12 minor keys such that C maj = 0, Db maj = 1 ... C min = 11, etc.

ALLOWED_KEYS = ['Db Major', 'D Major', 'Eb Major', 'E Major', 'F Major', 'Gb Major', 'G Major',
                'Ab Major', 'A Major', 'Bb Major', 'B Major', 'C minor', 'C# minor', 'D minor', 'Eb minor',
                'E minor', 'F minor', 'F# minor', 'G minor', 'G# minor', 'A minor', 'Bb minor', 'B minor']

SUBDIVISION = 8


# Test code
if os.path.exists('training_data.pkl'):
    with open('training_data.pkl', 'rb') as f:
        midis = pickle.load(f)

        for i, mid in enumerate(midis):
            print(f'Processing {i + 1} of {len(midis)}')
            try:
                arr = pretty_midi_to_numpy_array(mid, subdivision=SUBDIVISION, transpose_notes=True)
            except MIDIError as e:
                print(f'Error! {e}')

else:
    midis = filter_midi_files('nes_data', ALLOWED_TIME_SIGS, ALLOWED_KEYS, pickle_result=True)
