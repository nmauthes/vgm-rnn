'''

Utilities for parsing, modifying and filtering MIDI files. Also contains functions for converting from pretty_midi
to numpy array and vice versa.

:authors Nicolas Mauthes

'''

import os
import pickle
from copy import deepcopy

import pretty_midi
import numpy as np


ALLOWED_SUBDIVISIONS = [1, 2, 4, 8]
DEFAULT_VELOCITY = 110


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


def pretty_midi_to_numpy_array(midi_data, subdivision=4, max_duration=16, use_velocity=False, transpose_notes=False,
                               ignore_drums=True):
    '''
    Encodes a pretty_midi object into an array with shape (127, t, 2) where the first axis is MIDI pitch, t is
    the number of timesteps in the song, and the last axis is a pair [v, d] where v is the note's velocity (unless
    use_velocity is false) and d is its duration in number of timesteps.

    :param midi_data: The pretty_midi object to be encoded
    :param subdivision: The resolution at which to sample notes in the song where subdivision is the number of steps
    per quarter note(e.g. for subdivision=4, 1/subdivision represents a 16th note)
    :param max_duration: The maximum duration (number of steps) an encoded note can have. Default is 16.
    :param use_velocity: If true, velocity of the note is used for v (above), otherwise binary values where 1=on 0=off
    :param transpose_notes: If true, the notes will be transposed to C before encoding (key signature must be present)
    :param ignore_drums: If true, skips all drum instruments (i.e. where is_drum=True) in the song
    :return: A numpy array of shape (127, t, 2) encoding the notes in the pretty_midi object
    '''

    if subdivision not in ALLOWED_SUBDIVISIONS:
        raise MIDIError("That subdivision is not allowed!")

    if transpose_notes:
        try:
            transpose_to_c(midi_data)
        except:
            raise MIDIError('MIDI file could not be transposed.')

    step_size = midi_data.resolution // subdivision
    total_ticks = midi_data.time_to_tick(midi_data.get_end_time())

    if step_size == 0:
        raise MIDIError('The step size is too small (try decreasing the subdivision)')

    piano_roll = np.zeros((128, int(round(total_ticks / step_size)) + 1, 2), dtype=np.int)

    for inst in midi_data.instruments:
        if ignore_drums and inst.is_drum:
            continue

        for note in inst.notes:  # Notes that don't fall exactly on the grid are quantized to the nearest subdivision
            note_start = int(round(midi_data.time_to_tick(note.start) / step_size))
            duration = int(round(midi_data.time_to_tick(note.end - note.start) / step_size))

            if duration > max_duration:
                duration = max_duration

            piano_roll[note.pitch, note_start] = [note.velocity, duration] if use_velocity else [1, duration]

    return piano_roll


def numpy_array_to_pretty_midi(arr, subdivision=4, use_velocity=False, program=81, tempo=120, resolution=480):
    '''
    Decodes an array created using pretty_midi_to_numpy_array() and returns a pretty_midi object.

    :param arr: The numpy array to be decoded
    :param subdivision: The number of steps per quarter note. It is important that this has the same value as when
    the array was created in order to make sure note lengths are consistent.
    :param use_velocity: Whether or not velocity was used to encode the notes in the array. Should be the same as
    when created.
    :param program: The MIDI program number to use for playback. Default is 81 (Lead 1 (Square))
    :param tempo: The tempo of the pretty_midi object in BPM. Default is 120.
    :param resolution: The resolution of the pretty_midi object (i.e. ticks per quarter note)
    :return: A pretty_midi object based on the contents of the numpy array
    '''

    if subdivision not in ALLOWED_SUBDIVISIONS:
        raise MIDIError("That subdivision is not allowed!")

    step_size = resolution // subdivision

    mid = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=resolution)

    inst = pretty_midi.Instrument(program=program, is_drum=False)
    mid.instruments.append(inst)

    for pitch, time in np.ndindex(arr.shape[0], arr.shape[1]):
        (vel, dur) = arr[pitch][time][0], arr[pitch][time][1]

        if vel:
            note_start = time * step_size
            note = pretty_midi.Note(velocity=vel if use_velocity else DEFAULT_VELOCITY, pitch=pitch,
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


def transpose(midi_data, semitones):
    '''
    Transposes all the notes in a pretty_midi object by the specified number of semitones.
    Any drum instruments in the object will not be modified.

    :param midi_data: The pretty_midi object to be transposed
    :param semitones: The number semitones to transpose the notes up (positive) or down (negative)
    '''

    for inst in midi_data.instruments:
        if not inst.is_drum: # Don't transpose drum tracks
            for note in inst.notes:
                note.pitch += semitones


def transpose_to_c(midi_data):
    '''
    A special case of transpose that moves all notes to the key of C (major or minor)
    Note that in order to know the how many semitones to transpose, the original key (i.e. key_signature_changes[0]
    must be present. Otherwise an exception will be thrown.

    :param midi_data: The pretty_midi object to be transposed
    '''

    if midi_data.key_signature_changes:
        key = midi_data.key_signature_changes[0]
    else:
        raise MIDIError('MIDI key signature could not be determined.')

    pos_in_octave = key.key_number % 12

    if not pos_in_octave == 0:
        semitones = -pos_in_octave if pos_in_octave < 6 else 12 - pos_in_octave # Transpose up or down given dist from C

        transpose(midi_data, semitones)


def split_drums(midi_data):
    '''
    Splits a pretty_midi into two separate objects, one containing only the drum parts,
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
