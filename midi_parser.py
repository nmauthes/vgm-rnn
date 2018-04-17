'''

Utilities for parsing, modifying and filtering MIDI files. Also contains functions for converting from pretty_midi
to numpy array and vice versa.

:authors Nicolas Mauthes

'''

import os
import math
import pickle
from copy import deepcopy

import pretty_midi
import numpy as np


ALLOWED_SUBDIVISIONS = [1, 2, 4, 8]
DEFAULT_VELOCITY = 110


# Exception class for MIDI-specific errors
class MIDIError(Exception):
    pass


def filter_midi_files(data_folder, allowed_time_sigs, allowed_keys, max_time_changes=1, max_key_changes=1,
                      ignore_filters=False, pickle_result=False, path='training_data.pkl'):
    '''
    A function to filter a group of MIDI files by selecting only the ones that meet the specified criteria
    supplied for key, time signature, etc (e.g. only files in the key of C major with a 4/4 time signature).
    The files are returned as a list of pretty_midi objects. Note that the pickled file can get quite large
    when pickle_result is true.

    :param data_folder: The path of the folder containing the files to be filtered
    :param allowed_time_sigs: The time signatures to be allowed as an array of strings e.g. ['4/4', '3/4']
    :param allowed_keys: The key signatures to be allowed as an array of strings e.g. ['C Major', 'Bb Minor']
    :param max_time_changes: The maximum number of time signature changes allowed. Default is 1.
    :param max_key_changes: The maximum number of key signature changes allowed. Default is 1.
    :param ignore_filters: If true, all MIDI files in the folder will be converted regardless of filter settings
    :param pickle_result: If true, the resulting list of pretty_midi objects will be saved as a .pkl file
    :param path: The path where the .pkl file will be saved

    :return: A list of pretty_midi objects meeting the supplied filter settings
    '''

    midi_files = os.listdir(data_folder)

    errors = 0
    filtered_files = []
    for num, midi_file in enumerate(midi_files):
        print(f'Processing file {num + 1} of {len(midi_files)}')

        try:
            mid = pretty_midi.PrettyMIDI(os.path.join(data_folder, midi_file))

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
        print(f'{errors} files could not be parsed.')

    if pickle_result:
        print('Pickling results...')
        with open(path, 'wb') as f:
            pickle.dump(filtered_files, f)

    return filtered_files


def pretty_midi_to_piano_roll(mid, subdivision=4, max_duration=16, sensitivity=0.2, transpose_notes=False,
                              ignore_drums=True):

    '''
    Encodes a pretty_midi object into an piano-roll matrix with shape (t, 128) where the first axis is the number of
    timesteps and the second axis is MIDI pitch.

    :param mid: The pretty_midi object to be encoded
    :param subdivision: The resolution at which to sample notes in the song, where subdivision is the number of steps
    per quarter note(e.g. for subdivision=4, (1/subdivision) represents a 16th note)
    :param max_duration: The maximum duration (number of steps) an encoded note can have. Default is 16.
    :param sensitivity: This is a threshold that allows a window for notes that don't fall exactly on the metrical grid
    to be included. Specified as a percentage of step length. Useful for encoding e.g. live performances.
    :param transpose_notes: If true, the notes will be transposed to C before encoding (key signature must be present)
    :param ignore_drums: If true, skips all drum instruments (i.e. where is_drum=True) in the song

    :return: A numpy array of shape (t, 128) encoding the notes in the pretty_midi object
    '''

    if subdivision not in ALLOWED_SUBDIVISIONS:
        raise Exception('That subdivision is not allowed!')

    if not (0 <= sensitivity < 0.5):
        raise Exception('Sensitivity must be in the range [0, 0.5)')

    if transpose_notes:
        try:
            transpose_to_c(mid)
        except:
            raise MIDIError('MIDI file could not be transposed.')

    if mid.resolution % subdivision == 0: # Make sure we get even subdivisions of the quarter
        step_size = mid.resolution // subdivision
    else:
        raise MIDIError('Invalid step size (try changing the subdivision)')

    end_ticks = mid.time_to_tick(mid.get_end_time())
    num_measures = math.ceil(end_ticks / (mid.resolution * 4)) # Assumes 4/4 time

    piano_roll = np.zeros((num_measures * subdivision * 4, 128), dtype=np.int)

    for inst in mid.instruments:
        if ignore_drums and inst.is_drum:
            continue

        for note in inst.notes:
            relative_pos = mid.time_to_tick(note.start) / step_size
            nearest_step = round(relative_pos)

            # Ensure that notes don't jump between measures and prevent out of bounds errors
            if nearest_step % (subdivision * 4) == 0 and relative_pos < nearest_step:
                nearest_step -= 1

            # If note is in the right range, add it to the piano roll
            if nearest_step - sensitivity <= relative_pos <= nearest_step + sensitivity:
                note_start = int(nearest_step)
                duration = int(round(mid.time_to_tick(note.end - note.start) / step_size))

                if duration < 1:
                    duration = 1
                if duration > max_duration:
                    duration = max_duration

                piano_roll[note_start, note.pitch] = 1 # TODO ignoring duration for now

    return piano_roll


def piano_roll_to_pretty_midi(piano_roll, subdivision=4, program=82, tempo=120, resolution=480, pitch_offset=0):
    '''
    Decodes an array created using pretty_midi_to_numpy_array() and returns a pretty_midi object.

    :param piano_roll: The numpy array to be decoded
    :param subdivision: The number of steps per quarter note. It is important that this has the same value as when
    the array was created in order to make sure note lengths are consistent.
    :param program: The MIDI program number to use for playback. Default is 82 (Lead 1 (Square))
    :param tempo: The tempo of the pretty_midi object in BPM. Default is 120.
    :param resolution: The resolution of the pretty_midi object (i.e. ticks per quarter note)
    :param pitch_offset: Adds an offset to pitch indices, for use when the MIDI note range has been altered
    :return: A pretty_midi object based on the contents of the numpy array
    '''

    if subdivision not in ALLOWED_SUBDIVISIONS:
        raise Exception("That subdivision is not allowed!")

    step_size = resolution // subdivision

    mid = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=resolution)

    inst = pretty_midi.Instrument(program=program - 1, is_drum=False)
    mid.instruments.append(inst)

    for i, dur in np.ndenumerate(piano_roll):
        if dur:
            note_start = i[0] * step_size
            note = pretty_midi.Note(velocity=DEFAULT_VELOCITY, pitch=pitch_offset + i[1],
                                    start=mid.tick_to_time(note_start),
                                    end=mid.tick_to_time(int(note_start + step_size * dur)))

            mid.instruments[0].notes.append(note)

    return mid


def midi_file_to_pretty_midi(midi_file):
    '''
    Simple function for converting a raw MIDI file into pretty_midi format.

    :param midi_file: The MIDI file to be converted
    :return: A pretty_midi object
    '''

    if isinstance(midi_file, pretty_midi.PrettyMIDI):
        return midi_file

    try:
        mid = pretty_midi.PrettyMIDI(midi_file)
    except:
        raise MIDIError('Bad MIDI file!')

    return mid


def transpose(mid, semitones):
    '''
    Transposes all the notes in a pretty_midi object by the specified number of semitones.
    Any drum instruments in the object will not be modified.

    :param mid: The pretty_midi object to be transposed
    :param semitones: The number semitones to transpose the notes up (positive) or down (negative)
    '''

    for inst in mid.instruments:
        if not inst.is_drum: # Don't transpose drum tracks
            for note in inst.notes:
                note.pitch += semitones


def transpose_to_c(mid):
    '''
    A special case of transpose that moves all notes to the key of C (major or minor depending on original key)
    Note that in order to know the how many semitones to transpose, the original key (i.e. key_signature_changes[0]
    must be present. Otherwise an exception will be thrown.

    :param mid: The pretty_midi object to be transposed
    '''

    if mid.key_signature_changes:
        key = mid.key_signature_changes[0]
    else:
        raise MIDIError('MIDI key signature could not be determined.')

    pos_in_octave = key.key_number % 12

    if not pos_in_octave == 0:
        semitones = -pos_in_octave if pos_in_octave < 6 else 12 - pos_in_octave # Transpose up or down given dist from C

        transpose(mid, semitones)


def split_for_drums(mid):
    '''
    Splits a pretty_midi into two separate objects, one containing only the drum parts,
    and one containing all the other parts.

    :param mid: The pretty_midi object to be split
    :return: (drums, not_drums) Tuple containing pretty_midi objs split as described above
    '''

    drums = deepcopy(mid)
    not_drums = deepcopy(mid)

    drums.instruments = []
    not_drums.instruments = []

    for inst in mid.instruments:
        if inst.is_drum:
            drums.instruments.append(deepcopy(inst))
        else:
            not_drums.instruments.append(deepcopy(inst))

    return drums, not_drums
