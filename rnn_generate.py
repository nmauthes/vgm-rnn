'''

Use a trained model to predict note sequences.

:authors Nicolas Mauthes

'''

import os
import random
import argparse

import numpy as np

from rnn_train import *
from midi_parser import piano_roll_to_pretty_midi


# |---------- GENERATION PARAMS ----------|

GENERATED_FILE_PATH = 'examples'

GENERATED_STEPS = NOTES_IN_MEASURE * 8

# |---------------------------------------|


def sample(probabilities, threshold=0.2): # TODO make nicer
    for i, prob in np.ndenumerate(probabilities):
        if prob >= threshold:
            probabilities[i] = 1
        else:
            probabilities[i] = 0

    return probabilities


if __name__ == '__main__':
    # TODO load primer data
    # TODO load model weights
    # TODO predict outputs

    if os.path.exists(MIDI_DATA_PATH):
        midi_data = np.load(MIDI_DATA_PATH)
    else:
        raise Exception('MIDI data not found!')

    # Select a random sequence to prime prediction
    primer_index = random.randint(0, int(midi_data.shape[0] / SEQUENCE_LENGTH)) * SEQUENCE_LENGTH
    primer_sequence = [midi_data[primer_index:primer_index + SEQUENCE_LENGTH, MIN_MIDI_NOTE:MAX_MIDI_NOTE + 1]]
    primer_sequence = np.asarray(primer_sequence)

    model = build_model()
    probabilities = model.predict(primer_sequence)[0]

    piano_roll = sample(probabilities)

    generated_mid = piano_roll_to_pretty_midi(piano_roll, pitch_offset=MIN_MIDI_NOTE)
    generated_mid.write('gen.mid')