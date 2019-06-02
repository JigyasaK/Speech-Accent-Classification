# Speech Accent Classification

Everyone who speaks a language, speaks it with an accent. A particular accent essentially reflects a person's linguistic background. When people listen to someone speak with a different accent from their own, they notice the difference, and they may even make certain biased social judgments about the speaker.


## Getting Started

Two networks have been implemented - CNN and LSTM.

### Prerequisites

We have trained only three languages: `english`, `mandarin`, `arabic`

Packages you need

```
Keras
Librosa
pydub
sklearn
numpy
pandas
```

### How to run

Steps to run the project

First we need to download the required audio files from the speech accent archive database which is located here - accent.gmu.edu/browse_language.php

Create a folder inside `data` folder named as `audio`.

Go to `code` folder. And then run the `getaudio.py` file. 
```
cd code
python getaudio.py
```
Wait till all the required audio is downloaded

Now, lets start training and predicting.
If you want to run train with CNN, enter the following :
```
python main.py cnn 100
```
Where 100 is the number of epochs. 

And, if you want to train with LSTM, enter the following:
```
python main.py lstm 100
```

`NOTE: It is necessary to specify the network you want to train. However, if epochs are not specified, default will be 10.`



