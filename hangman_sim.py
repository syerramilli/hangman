import random
from typing import List, Dict

class HangmanSimulator:
    '''
    This class is used to simulate the Hangman Game. The player has to 
    guess the word that the computer has chosen. In each turn, the player
    can guess a letter. If the letter is in the word, all instances of the
    letter will be revealed. If the letter is not in the word, the player
    will lose a try. The player has a maximum of 6 tries to guess the word.

    The game can be played by calling the `start_game` method. The player can
    then guess the word by calling the `play` method. The game can be reset
    by calling the `start_game` method again.

    Args:
        dictionary (List[str]): A list of words that the computer can choose
            from. The computer will choose a word at random from this list 
            for each game.
    '''
    def __init__(self, dictionary: List[str]):

        self.max_guesses = 6
        self.dictionary = dictionary
        
        # initiate game specific variables
        self.game_id = -1
        self.tries_remains = self.max_guesses
        self.guessed = []
    
    def choose_word(self):
        '''
        This method is used to choose a word from the dictionary
        '''
        return random.choice(self.dictionary)
    
    def display_word(self, word, guessed):
        '''
        This method is used to display the word with the letters that the 
        player has guessed
        '''
        return ''.join([f"{c if c in guessed else '_'} " for c in word])
    
    def start_game(self) -> Dict:
        '''
        Begin a new round of the game
        '''
        if len(self.dictionary) == 0:
            raise RuntimeError("Dictionary is empty. Please populate the dictionary first")

        self.game_id += 1
        self.word = self.choose_word()
        self.rem_chars = set(self.word)
        self.tries_remains = self.max_guesses
        self.guessed = []
        self.status = 'ongoing'

        return {
            'game_id': self.game_id,
            'status': self.status,
            'tries_remains': self.tries_remains,
            'word': self.display_word(self.word, self.guessed)
        }
    
    def play(self, guess_letter:str) -> Dict:
        '''
        The player can guess a letter in the word. This method will return the
        status of the game after the player has guessed the letter.

        Args:
            guess_letter (str): The letter that the player has guessed

        Returns:
            dict: A dictionary containing the status of the game after the player
                has guessed the letter. The dictionary will contain the following
                keys:
                    - status (str): The status of the game. It can be one of the
                        following values:
                            - 'ongoing': The game is still ongoing
                            - 'success': The player has successfully guessed the word
                            - 'failed': The player has failed to guess the word
                    - tries_remains (int): The number of tries that the player has
                        remaining
                    - word (str): The word with letters that the player has guessed 
                        revealed
        '''
        if self.status != 'ongoing':
            if len(self.rem_chars) == 0:
                return {
                    'status': self.status,
                    'tries_remains': self.tries_remains,
                    'reason': 'Player has already won'
                }
            
            if self.tries_remains == 0:
                return {
                    'status': self.status,
                    'tries_remains': self.tries_remains,
                    'reason': 'Player has already lost'
                }
        
        # ongoing
        if guess_letter in self.rem_chars:
            self.rem_chars.remove(guess_letter)
            self.guessed.append(guess_letter)
            display_word = self.display_word(self.word, self.guessed)
            out = {
                'status': 'ongoing',
                'tries_remains': self.tries_remains,
                'word': display_word
            }

            if len(self.rem_chars) == 0:
                self.status = 'success'
                out['status'] = self.status
                out['reason'] = 'Player has won'

        else:
            self.tries_remains -= 1
            out = {
                'status': 'ongoing',
                'tries_remains': self.tries_remains,
                'word': self.display_word(self.word, self.guessed)
            }
            if self.tries_remains == 0:
                self.status = 'failed'
                out['status'] = self.status
                out['reason'] = '# of tries exceeded!'

        return out   