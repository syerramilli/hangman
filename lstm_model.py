import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List

def letter_to_index(letter: str) -> int:
    return 27 if letter == '_' else ord(letter) - 97 + 1

def index_to_letter(index: int) -> str:
    return '_' if index == 27 else chr(index + 96)

class MaskedLettersDataset(Dataset):
    '''
    Dataset class for the masked letters task. In each sample, a word is
    selected from the list of words. A random subset of the unique
    characters in the word is selected and masked with an underscore. 
    '''
    def __init__(self, words_list: List[str]) -> None:
        super().__init__()
        self.words_list = words_list
    
    def __len__(self) -> int:
        return len(self.words_list)
    
    def __getitem__(self, idx):
        word = self.words_list[idx]
        unique_letters = list(set(word))
        if len(unique_letters) == 1:
            num_letters_to_mask = 1
        else:
            num_letters_to_mask = random.randint(1, len(unique_letters) - 1)

        # randomly select unique characters to mask
        chars_to_mask = random.sample(unique_letters, num_letters_to_mask)

        # mask all occurrences of the selected characters
        label_word = list(word)
        masked_word = []
        mask_positions = [0] * len(word)

        for i, c in enumerate(word):
            if c in chars_to_mask:
                masked_word.append('_')
                mask_positions[i] = 1
            else:
                masked_word.append(c)

        return masked_word, label_word, mask_positions

def collate_fn(batch):
    masked_words, label_words, mask_positions = zip(*batch)
    try:
        masked_words_padded = pad_sequence([
            torch.tensor([letter_to_index(c) for c in word]) for word in masked_words
            ], batch_first=True
        )

    except Exception as e:
        print(masked_words)
        raise e
    
    label_words_padded = pad_sequence([
        torch.tensor([
            letter_to_index(c) for c in word
        ]) for word in label_words], 
        batch_first=True
    )
    mask_positions_padded = pad_sequence([
        torch.tensor(mask) for mask in mask_positions
    ], batch_first=True)

    return masked_words_padded, label_words_padded, mask_positions_padded

class CharBiLSTM(nn.Module):
    '''
    BiLSTM model for predicting the probability of each character in a word given 
    the entire context. The architecture consists of an embedding layer for the 
    characters, followed by one or more bidirectional LSTM layer, and finally a 
    fully connected layer to predict the logits for each character in the word. 

    Args:
        vocab_size (int): The number of unique characters in the vocabulary
        embedding_dim (int): The dimension of the character embeddings
        hidden_dim (int): The dimension of the hidden states in the LSTM layers
        num_layers (int): The number of LSTM layers in the model. (Default: 2)
    '''
    def __init__(self, vocab_size:int, embedding_dim:int, hidden_dim:int, num_layers:int=2) -> None:
        super(CharBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, 
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        embedding = self.embedding(x)
        lstm_out, _ = self.lstm(embedding)
        logits = self.fc(lstm_out)
        return logits