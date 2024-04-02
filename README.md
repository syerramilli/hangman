# Playing Hangman using LSTM models

## Hangman Game

Hangman is a word guessing game. The player is given a word with letters masked. At each turn, the player guesses a letter, and if the letter is present in the word, all instances of the letter are revealed. If the letter is not present in the word, the player loses a life. The player has a fixed number (6 here) of lives to guess the word. The player wins if they can guess the word before running out of lives.

## Approach

The idea here is to use a character level LSTM model that predicts the probabilities of the 26 letters at each position in the word. During the game, the model is used to predict these probabilities at the masked positions. Given these predictions, the probability of a letter in the word is computed as the product of the probabilities of the letter at each mask position. The letter with the highest probability is chosen as the next guess. 

## LSTM Model and Training

The architecture of the LSTM model consists of an embedding layers (for the characters) followed by one or more bidirectional LSTM layers, and finally a dense layer with a softmax activation for each position in the word. There are 26 output nodes corresponding to the 26 letters of the alphabet. 

Since the model will be used to predict the probabilities of the letters at the masked positions, we will use masked language modeling to train the model. The training data consists of dictionary of words from the NLTK words corpus. For each word in the training data, we select one or more letters and mask all occurences of these letters in the word. The objective then is to predict the letters at the masked positions given the context of the word - we use categorical crossentropy loss for this. Rather than pre-generating the masked words, we use a custom data generator to generate the masked words on the fly. 