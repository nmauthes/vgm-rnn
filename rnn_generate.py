'''

Use a trained model to predict note sequences.

:authors Nicolas Mauthes

'''

import os
import random
import argparse
import pickle

import numpy as np

from rnn_train import MIDI_DATA_PATH, SUBDIVISION
from rnn_train import MODEL_FOLDER, SAVED_WEIGHTS_PATH, SAVED_DICT_PATH, build_model, split_xy, truncate_data
from rnn_train import SEQUENCE_LENGTH, MIN_MIDI_NOTE, MAX_MIDI_NOTE, MIDI_NOTE_RANGE, TRUNCATE_DATA, TRUNCATE_PERCENT
from midi_parser import stringify, unstringify, piano_roll_to_pretty_midi


# |---------- GENERATION PARAMS ----------|

MIDI_PROGRAM = 82

GENERATED_MIDI_FOLDER = 'examples'
GENERATED_FILENAME = 'example.mid'

SAVE_PRIMER_SEQUENCE = False # TODO check behavior
PRIMER_FILENAME = 'primer.mid'

NUM_ITERATIONS = SEQUENCE_LENGTH
SAMPLING_THRESHOLD = 0.35

# |---------------------------------------|


def generate_sequence(model, chord_dict, primer, seq_length, num_steps): # TODO seq_length
    generated = []

    # Prepare primer sequence input
    seq_in = stringify(primer)
    seq_in = [chord_dict[chord] for chord in seq_in]
    seq_in = np.asarray([seq_in])

    # Flip keys/values
    chord_dict = {i: chord for chord, i in chord_dict.items()}

    for _ in range(num_steps):
        predicted = model.predict_classes(seq_in)

        generated.append(chord_dict[predicted[0]])

        next_seq = np.append(seq_in[0, 1:], predicted)
        seq_in = np.asarray([next_seq])



    generated = np.asarray(generated)
    generated = unstringify(generated, MIDI_NOTE_RANGE)

    return generated


# |---------- COMMAND LINE ARGS ----------|

parser = argparse.ArgumentParser(
    description='Generate new MIDI files using a trained model',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='See the README for further questions'
)

parser.add_argument(
    '--generated_filename',
    default=GENERATED_FILENAME,
    help='Name of the generated MIDI file.'
)
parser.add_argument(
    '--saved_weights_path',
    default=SAVED_WEIGHTS_PATH,
    help='The path to the saved weights to use for prediction.'
)
parser.add_argument(
    '--save_primer',
    type=bool,
    default=SAVE_PRIMER_SEQUENCE,
    help='Whether to save the primer sequence or not.'
)

# |---------------------------------------|


if __name__ == '__main__':
    args = parser.parse_args()

    if os.path.exists(MIDI_DATA_PATH):
        midi_data = np.load(MIDI_DATA_PATH)
    else:
        raise Exception('MIDI data not found!')

    if TRUNCATE_DATA:
        midi_data = truncate_data(midi_data, TRUNCATE_PERCENT)

    # Select a random sequence to prime prediction
    primer_index = random.randint(0, int(midi_data.shape[0] / SEQUENCE_LENGTH)) * SEQUENCE_LENGTH
    primer_sequence = midi_data[primer_index:primer_index + SEQUENCE_LENGTH, MIN_MIDI_NOTE:MAX_MIDI_NOTE + 1]
    primer_sequence = np.asarray(primer_sequence)

    # Load saved chord dict
    if os.path.exists(SAVED_DICT_PATH):
        with open(SAVED_DICT_PATH, 'rb') as f:
            chord_dict = pickle.load(f)
    else:
        raise Exception('Chord dictionary not found!')

    vocab_size = len(chord_dict)

    # Use model to predict the next sequence given primer
    model = build_model(vocab_size)
    model.load_weights(os.path.join(MODEL_FOLDER, args.saved_weights_path))

    piano_roll = generate_sequence(model, chord_dict, primer_sequence, SEQUENCE_LENGTH, NUM_ITERATIONS)

    generated_mid = piano_roll_to_pretty_midi(piano_roll, subdivision=SUBDIVISION, program=MIDI_PROGRAM,
                                              pitch_offset=MIN_MIDI_NOTE)
    generated_mid.write(os.path.join(GENERATED_MIDI_FOLDER, args.generated_filename))

    # Save the primer sequence for reference
    if args.save_primer:
        primer = piano_roll_to_pretty_midi(primer_sequence[0], subdivision=SUBDIVISION, program=MIDI_PROGRAM,
                                           pitch_offset=MIN_MIDI_NOTE)
        primer.write(os.path.join(GENERATED_MIDI_FOLDER, PRIMER_FILENAME))
        print(f'Primer saved as \'{PRIMER_FILENAME}\'')

    print(f'Generated file saved as \'{args.generated_filename}\'')
