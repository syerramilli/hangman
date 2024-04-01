import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from typing import List
from tqdm import tqdm
import toml

def letter_to_index(letter: str) -> int:
    return 27 if letter == '_' else ord(letter) - 97 + 1

def index_to_letter(index: int) -> str:
    return '_' if index == 27 else chr(index + 96)


class MaskedLettersDataset(Dataset):
    def __init__(self, words_list) -> None:
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
        masked_words_padded = pad_sequence([torch.tensor([letter_to_index(c) for c in word]) for word in masked_words], batch_first=True)
    except Exception as e:
        print(masked_words)
        raise e
    label_words_padded = pad_sequence([torch.tensor([letter_to_index(c) for c in word]) for word in label_words], batch_first=True)
    mask_positions_padded = pad_sequence([torch.tensor(mask) for mask in mask_positions], batch_first=True)
    return masked_words_padded, label_words_padded, mask_positions_padded

class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2) -> None:
        super(CharBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        embedding = self.embedding(x)
        lstm_out, _ = self.lstm(embedding)
        logits = self.fc(lstm_out)
        return logits

if __name__ == '__main__':
    # Load the settings from the config file
    config = toml.load("lstm_config.toml")

    EMBEDDING_DIM = config["model"]["embedding_dim"]
    HIDDEN_DIM = config["model"]["hidden_dim"]

    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = config["training"]["epochs"]
    LEARNING_RATE = config["training"]["learning_rate"]
    SPLIT_RATIO = config["vocab"]["split_ratio"]
    SPLIT_DICTIONARY = config["vocab"]["split_dictionary"]

    with open('words_250000_train.txt') as f:
        dictionary = f.read().splitlines()

    print('Creating the dataset and the dataloader')
    if SPLIT_DICTIONARY:
        dictionary, _ = train_test_split(dictionary, train_size=SPLIT_RATIO, random_state=42)

    dataset = MaskedLettersDataset(dictionary)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # instantiate the model
    print('Instantiating the model')
    model = CharBiLSTM(28, EMBEDDING_DIM, HIDDEN_DIM)
    
    # define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    print('Training the model')
    for epoch in range(EPOCHS):
        with tqdm(dataloader, desc=f'Epoch {epoch}', unit='batch') as t:
            for i, (masked_words, label_words, mask_positions) in enumerate(t):
                optimizer.zero_grad()
                logits = model(masked_words)
                loss = criterion(logits[mask_positions == 1], label_words[mask_positions == 1])
                loss.backward()
                optimizer.step()
                #if i % 100 == 0:
                t.set_postfix({'Loss': loss.item()})
                t.update(1)

    # save the model
    print('Saving the model')
    torch.save(model.state_dict(), config['saving']['filename'])