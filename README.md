# VGM-RNN
A recurrent neural network for generating video game music.

## How to use

To run the code, you will need Tensorflow, Keras and a few other libraries (see requirements.txt)

### Training the network

To train the network, navigate to the folder where you cloned the repo, then run the following command:

```
python rnn_train.py
```

Note that the weights provided with the repo will be overwritten unless you rename them.

### Generating new MIDI files

You can use the trained weights provided with this repo or your own weights to generate new MIDI sequences. To generate new MIDI sequences using the model:

```
python rnn_generate.py
```
