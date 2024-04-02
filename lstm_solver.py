import torch
import toml
from lstm_model import CharBiLSTM, letter_to_index as torch_letter_to_index
from collections import defaultdict
from hangman_sim import HangmanSimulator
from typing import List, Dict

def letter_to_index(letter):
    return ord(letter) - ord('a')

def index_to_letter(idx):
    return chr(idx + ord('a'))

class HangmanLSTMSolver:
    '''
    Hangman solver using a LSTM model to predict the most probable letter at each turn. 
    
    A pre-trained (bidirectional) LSTM model is first used to compute the probability of each letter 
    (that is not guessed so fat) at each masked position in the word given the entire word. The total 
    probability of a letter is then given by the product of its probabilities at all masked positions. 
    The letter with the highest total probability is then guessed. 
    
    If all letters are masked, the unigram probabilities are used to guess the next letter. The unigram
    probabilities are precomputed based on the supplied `dictionary` (a list of words). This is not 
    required to be the dictionary used by the hangman simulator. Separate unigram probabilities are
    computed for each word length.

    Args:
        - simulator: HangmanSimulator object
        - dictionary: list of words to compute unigram probabilities
        - config_file: path to the configuration file for the LSTM model
    '''
    def __init__(self, 
        simulator: HangmanSimulator, 
        dictionary: List[str], 
        config_file: str
    ):
        self.simulator = simulator
        self.guessed_letters = []
        self.full_dictionary = None

        # counts - for unigrams,
        self.unigrams_by_length = defaultdict(lambda: [0] * 26)
        self.unigrams = [0] * 26
        self.populate_cts(dictionary)

        config = toml.load(config_file)
        self.model = CharBiLSTM(
            28, 
            config['model']['embedding_dim'], 
            config['model']['hidden_dim'],
            config['model']['num_layers']
        )
        self.model.load_state_dict(torch.load(config['saving']['model_state']))
    
    def populate_cts(self, dictionary):
        for word in dictionary:
            word_len = len(word)

            # count unigrms
            for i, letter in enumerate(word):
                let_idx = letter_to_index(letter)
                self.unigrams_by_length[word_len][let_idx] += 1

    def play_the_game(self, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary

        response = self.simulator.start_game()
        game_id = response.get('game_id')
        word = response.get('word')
        tries_remains = response.get('tries_remains')

        if verbose:
            print("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id, tries_remains, word))
        
        while tries_remains > 0:
            guess_letter = self.guess(word)
            self.guessed_letters.append(guess_letter)

            if verbose:
                    print("Guessing letter: {0}".format(guess_letter))

            res = self.simulator.play(guess_letter)
            status = res.get('status')
            tries_remains = res.get('tries_remains')
            if status=="success":
                if verbose:
                    print("Successfully finished game {0} with word {1}".format(game_id, res.get('word')[::2]))
                return res

            elif status=="failed":
                reason = res.get('reason', '# of tries exceeded!')
                if verbose:
                    print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                return res
            elif status=="ongoing":
                word = res.get('word')

        return res

    def guess(self, word):
        # clean the word so that we strip away the space characters
        word = word[::2]
        
        # get the number of masked latters
        masked_ct = word.count('_')
        if masked_ct == len(word):
            # all letter are masked - use unigram probabilities
            probs = self.unigram_probs(word)
        else:
            # some letters are filled - use the model to predict the next letter
            word_tensor = torch.tensor([torch_letter_to_index(c) for c in word]).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(word_tensor).squeeze(0)[:, 1:27]
            
            mask = torch.tensor([1.0 if c == '_' else 0.0 for c in word])
            masked_logits = logits * mask.unsqueeze(-1)
            summed_logits = masked_logits.sum(0) / mask.sum()
            probs = torch.softmax(summed_logits, 0).numpy()

            # mask = torch.tensor([True if c == '_' else False for c in word])
            # max_logits = logits[mask,:].max(0).values
            # probs = torch.softmax(max_logits, 0).numpy()
            
        
        # find the most probable letter
        max_prob = 0
        guess_letter = '!'
        for i in range(26):
            if index_to_letter(i) in self.guessed_letters:
                continue

            if probs[i] > max_prob:
                max_prob = probs[i]
                guess_letter = chr(i + ord('a'))

        if guess_letter == '!':
            raise RuntimeError(f"No guess letter found for word {word} with guessed letters {self.guessed_letters}")
        
        return guess_letter

    def unigram_probs(self, word):
        total_count = 0
        letter_count = [0] * 26
        len_word = len(word)

        for letter in word:
            if letter != '_':
                # character is filled
                continue
            
            for i in range(26):
                if index_to_letter(i) in self.guessed_letters:
                    continue
                
                unigram_cts = self.unigrams if len_word not in self.unigrams_by_length else self.unigrams_by_length[len_word]
                total_count += unigram_cts[i]
                letter_count[i] += unigram_cts[i]
        
        probs = [0] * 26
        for i in range(26):
            if index_to_letter(i) not in self.guessed_letters:
                probs[i] = letter_count[i] / total_count

        return probs