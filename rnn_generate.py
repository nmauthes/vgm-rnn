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

MIDI_PROGRAM = 82

GENERATED_MIDI_FOLDER = 'examples'
GENERATED_NAME = 'example'

PRIMER_NAME = 'primer'

NUM_GENERATED = 5
SAMPLING_THRESHOLD = 0.35

# |---------------------------------------|


def prob_matrix_to_piano_roll(prob_matrix, threshold=0.2):
    ''' Produce a piano-roll by sampling from a probability matrix. '''

    for i, prob in np.ndenumerate(prob_matrix):
        if prob >= threshold:
            prob_matrix[i] = 1
        else:
            prob_matrix[i] = 0

    return prob_matrix


# |---------- COMMAND LINE ARGS ----------|

parser = argparse.ArgumentParser(
    description='Generate new MIDI files using a trained model',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='See the README for further questions'
)

parser.add_argument(
    '--generated_name',
    default=GENERATED_NAME,
    help='Name to give the generated MIDI file(s).'
)
parser.add_argument(
    '--num_generated',
    type=int,
    default=NUM_GENERATED,
    help='Number of MIDI files to generate.'
)
parser.add_argument(
    '--save_primer',
    action='store_true',
    help='Whether to save the primer sequence or not.'
)
parser.add_argument(
    '--sampling_threshold',
    type=float,
    default=SAMPLING_THRESHOLD,
    help='The threshold value to use when sampling from prob matrix.'
)
parser.add_argument(
    '--saved_weights_path',
    default=SAVED_WEIGHTS_PATH,
    help='The path to the saved weights to use for prediction.'
)

# |---------------------------------------|


if __name__ == '__main__':
    args = parser.parse_args()

    if os.path.exists(MIDI_DATA_PATH):
        midi_data = np.load(MIDI_DATA_PATH)
    else:
        raise Exception('MIDI data not found!')

    print('-' * 25)
    print('Generating MIDI files...')
    print('-' * 25)

    for i in range(args.num_generated):
        # Select a random sequence to prime prediction
        primer_index = random.randint(0, int(midi_data.shape[0] / SEQUENCE_LENGTH)) * SEQUENCE_LENGTH
        primer_sequence = [midi_data[primer_index:primer_index + SEQUENCE_LENGTH, MIN_MIDI_NOTE:MAX_MIDI_NOTE + 1]]
        primer_sequence = np.asarray(primer_sequence)

        # Build the model and load weights
        model = build_model()
        model.load_weights(os.path.join(MODEL_FOLDER, args.saved_weights_path))

        # Use model to predict the next sequence given primer
        note_probs = model.predict(primer_sequence)[0]

        # Convert prob matrix to piano-roll and save
        piano_roll = prob_matrix_to_piano_roll(note_probs, threshold=args.sampling_threshold)
        generated_mid = piano_roll_to_pretty_midi(piano_roll, subdivision=SUBDIVISION, program=MIDI_PROGRAM,
                                                  pitch_offset=MIN_MIDI_NOTE)
        generated_mid.write(os.path.join(GENERATED_MIDI_FOLDER, args.generated_name + f'_{i + 1}.mid'))

        # Save the primer sequence for reference
        if args.save_primer:
            primer = piano_roll_to_pretty_midi(primer_sequence[0], subdivision=SUBDIVISION, program=MIDI_PROGRAM,
                                               pitch_offset=MIN_MIDI_NOTE)
            primer.write(os.path.join(GENERATED_MIDI_FOLDER, args.generated_name + f'_{i + 1}' + '_' + PRIMER_NAME + '.mid'))

    print(f'Generated file(s) saved in folder \'{GENERATED_MIDI_FOLDER}\'')