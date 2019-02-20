# VGM-RNN

A recurrent neural network for generating video game music.

<img src="vgm_rnn.pn">

Read the paper: http://scholarworks.sjsu.edu/etd_projects/595/

Take the survey: https://goo.gl/forms/78UI3FTFjHOx5Oq82

## How to use

To run the code, you will need Tensorflow, Keras and a few other libraries (see requirements.txt)

### Training the network

To train the network, navigate to the folder where you cloned the repo, then run the following command:

```
python rnn_train.py
```

### Generating new MIDI files

You can use the trained weights to generate new MIDI sequences using the model. After training the model, run this command:

```
python rnn_generate.py
```
