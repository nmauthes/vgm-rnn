'''

Use a trained model to predict note sequences.

:authors Nicolas Mauthes

'''

import os
import random
import argparse

import numpy as np

from rnn_train import MIDI_DATA_PATH, SUBDIVISION
from rnn_train import MODEL_FOLDER, SAVED_WEIGHTS_PATH, SEQUENCE_LENGTH, MIN_MIDI_NOTE, MAX_MIDI_NOTE, build_model
from midi_parser import piano_roll_to_pretty_midi


# |---------- GENERATION PARAMS ----------|

MIDI_PROGRAM = 1

GENERATED_MIDI_FOLDER = 'examples'
GENERATED_FILENAME = 'example.mid'

SAVE_PRIMER_SEQUENCE = True
PRIMER_FILENAME = 'primer.mid'

NUM_ITERATIONS = 1
SAMPLING_THRESHOLD = 0.35

# |---------------------------------------|


def prob_matrix_to_piano_roll(prob_matrix, threshold=0.2): # TODO make nicer
    for i, prob in np.ndenumerate(prob_matrix):
        if prob >= threshold:
            prob_matrix[i] = 1
        else:
            prob_matrix[i] = 0

    return prob_matrix


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
    model.load_weights(os.path.join(MODEL_FOLDER, SAVED_WEIGHTS_PATH))

    note_probs = model.predict(primer_sequence)[0]

    # Save the primer sequence for reference
    if SAVE_PRIMER_SEQUENCE:
        primer = piano_roll_to_pretty_midi(primer_sequence[0], subdivision=SUBDIVISION, program=MIDI_PROGRAM,
                                           pitch_offset=MIN_MIDI_NOTE)
        primer.write(os.path.join(GENERATED_MIDI_FOLDER, PRIMER_FILENAME))
        print(f'Primer saved as \'{PRIMER_FILENAME}\'')

    piano_roll = prob_matrix_to_piano_roll(note_probs, threshold=SAMPLING_THRESHOLD)
    generated_mid = piano_roll_to_pretty_midi(piano_roll, subdivision=SUBDIVISION, program=MIDI_PROGRAM,
                                              pitch_offset=MIN_MIDI_NOTE)
    generated_mid.write(os.path.join(GENERATED_MIDI_FOLDER, GENERATED_FILENAME))

    print(f'Generated file saved as \'{GENERATED_FILENAME}\'')