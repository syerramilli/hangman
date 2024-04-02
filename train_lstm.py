import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from lstm_model import MaskedLettersDataset, collate_fn, CharBiLSTM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import toml

if __name__ == '__main__':
    # Load the settings from the config file
    config = toml.load("lstm_config.toml")

    EMBEDDING_DIM = config["model"]["embedding_dim"]
    HIDDEN_DIM = config["model"]["hidden_dim"]
    NUM_LAYERS = config["model"]["num_layers"]

    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = config["training"]["epochs"]
    LEARNING_RATE = config["training"]["learning_rate"]

    FILE = config["vocab"]["file"]
    SPLIT_RATIO = config["vocab"]["split_ratio"]
    SPLIT_DICTIONARY = config["vocab"]["split_dictionary"]

    with open(FILE) as f:
        dictionary = f.read().splitlines()

    print('Creating the dataset and the dataloader')
    if SPLIT_DICTIONARY:
        dictionary, _ = train_test_split(dictionary, train_size=SPLIT_RATIO, random_state=42)

    dataset = MaskedLettersDataset(dictionary)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # instantiate the model
    print('Instantiating the model')
    # the vocab size is 28 because we have 26 letters and 2 special characters
    # 0 for padding and 27 for masked characters
    model = CharBiLSTM(28, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    
    # define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # training losses
    training_losses = []

    print('Training the model')
    for epoch in range(EPOCHS):
        with tqdm(dataloader, desc=f'Epoch {epoch}', unit='batch') as t:
            for i, (masked_words, label_words, mask_positions) in enumerate(t):
                optimizer.zero_grad()
                # Transfer data to device
                masked_words = masked_words.to(device)
                label_words = label_words.to(device)
                mask_positions = mask_positions.to(device)
                
                # Forward pass
                logits = model(masked_words)
                loss = criterion(logits[mask_positions == 1], label_words[mask_positions == 1])

                # Backward pass
                loss.backward()
                optimizer.step()
                #if i % 100 == 0:
                t.set_postfix({'Loss': loss.item()})
                t.update(1)
            
            training_losses.append(loss.item())

    # save the model
    print('Saving the model')
    torch.save(model.state_dict(), config['saving']['model_state'])

    with open(config['saving']['training_loss_path'], 'w') as f:
        for loss in training_losses:
            f.write(f'{loss}\n')