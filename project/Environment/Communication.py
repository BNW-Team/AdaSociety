import numpy as np
from typing import Sequence, Mapping

class CommunicationBoard:
    def __init__(self, player_num: int, dim: int, player_name2id: Mapping[str, int]) -> None:
        """The board that stores the latest message

        Args:
            player_num: the number of players
            dim: communication words dimension
            player_name2id: A dictionary that takes players' ids as keys, and players' names
                as values

        Attributes:
            player_num: the number of players
            dim: communication words dimension
            board: the communication board that stores the latest message
            player_name2id: A dictionary that takes players' ids as keys, and players' names
                as values
        """
        self.player_num = player_num
        self.dim = dim
        self.board = np.zeros((player_num, player_num, dim))
        self.player_name2id = player_name2id
        
    def clear(self):
        """Clear the communication board to blank"""
        self.board = np.zeros((self.player_num, self.player_num, self.dim))

    def load_message(self, _from: int, _to: Sequence[str], content: np.array):
        '''Loading message from players' actions to communication board
        
        Args:
            _from: Player id who sends the message
            _to: name(s) of Player(s) that the message aims at
            content: Communication contents in ``self.dim`` length
        '''
        
        assert len(content) == self.dim, f'Communication size should be dim = {self.dim}'
        _to_id = [self.player_name2id[name] for name in _to]
        self.board[_from][_to_id] = content