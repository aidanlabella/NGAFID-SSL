import torch
from torch.utils.data import Dataset


class ScorePairDataset(Dataset):
    def __init__(self, pairs, flight_data=None):
        self.pairs = pairs
        self.flight_data = None
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        pair = self.dataset["Positive Pairs"][index]
        flight_pair = pair
        # flight_pair = (self.flight_data[pair[0]], self.flight_data[pair[1]])
        return flight_pair, -1

        

