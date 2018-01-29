# TODO Add docstrings

'''

Utilities for parsing, modifying and filtering MIDI files.

'''

import os
import pickle
from copy import deepcopy

import pretty_midi
import numpy


# TODO Allow for multiple data folders?
def filter_midi_files(data_folder, allowed_time_sigs, allowed_keys, max_time_changes=1, max_key_changes=1,
                      ignore_filters=False, pickle_result=False, path='training_data.pkl'):

    midi_files = os.listdir(data_folder)

    errors = 0
    filtered_files = []
    for num, midi_file in enumerate(midi_files):
        print(f'Processing file {num + 1} of {len(midi_files)}')

        try: # TODO check logic
            mid = pretty_midi.PrettyMIDI(os.path.join(data_folder, midi_file)) # Try/except

            if not ignore_filters:
                time_sig_is_good, key_is_good = False, False

                if mid.time_signature_changes and len(mid.time_signature_changes) <= max_time_changes:
                    time_sig_is_good = all(f'{ts.numerator}/{ts.denominator}' in allowed_time_sigs for ts in mid.time_signature_changes)

                if mid.key_signature_changes and len(mid.key_signature_changes) <= max_key_changes:
                    key_is_good = all(pretty_midi.key_number_to_key_name(key.key_number) in allowed_keys for key in mid.key_signature_changes)

                if time_sig_is_good and key_is_good:
                    filtered_files.append(mid)

            else:
                filtered_files.append(mid)
        except:
            errors += 1

    print(f'{len(filtered_files)} MIDI files found.')

    if errors:
        print(f'{errors} files could not be parsed')

    if pickle_result:
        print('Pickling results...')
        with open(path, 'wb') as f:
            pickle.dump(filtered_files, f)

    return filtered_files



def pretty_midi_to_numpy_array(midi_data):
    pass


# TODO Get key function
# TODO Quantize function

def _midi_file_to_pretty_midi(midi_data):
    if isinstance(midi_data, pretty_midi.PrettyMIDI):
        return midi_data

    try:
        mid = pretty_midi.PrettyMIDI(midi_data)
    except:
        raise Exception('Bad MIDI file!')

    return mid


def _transpose(midi_data, semitones):
    for inst in midi_data.instruments:
        if not inst.is_drum: # Don't transpose drum tracks
            for note in inst.notes:
                note.pitch += semitones


def _split_drums(midi_data):
    '''
    Split a pretty_midi into two separate objects, one containing only the drum parts,
    and one containing all the other parts.

    :param midi_data: The pretty_midi object to be split
    :return: (drums, not_drums) Tuple containing pretty_midi objs split as described above
    '''

    drums = deepcopy(midi_data)
    not_drums = deepcopy(midi_data)

    drums.instruments = []
    not_drums.instruments = []

    for inst in midi_data.instruments:
        if inst.is_drum:
            drums.instruments.append(deepcopy(inst))
        else:
            not_drums.instruments.append(deepcopy(inst))

    return drums, not_drums