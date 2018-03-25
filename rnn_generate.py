'''

Use a trained model to predict note sequences.

'''


from midi_parser import piano_roll_to_pretty_midi
from rnn_train import MIN_MIDI_NOTE, MIDI_DATA_PATH


GENERATED_FILE_PATH = 'examples'