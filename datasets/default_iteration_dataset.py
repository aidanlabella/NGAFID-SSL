import torch
import pandas as pd
from torch.utils.data import Dataset

class DefaultIterationDataset(Dataset):
    def __init__(self, flight_id_topath):
        self.flight_id_topath = flight_id_topath.reset_index()
    
    def __len__(self):
        return len(self.flight_id_topath)
    
    def __getitem__(self, index):
        path = self.flight_id_topath.loc[index, "file_path"]
        flight_id = self.flight_id_topath.loc[index, "flight_id"]
        flight = pd.read_csv(path, na_values=[' NaN', 'NaN', 'NaN '])
        flight_T = flight.T
        flight_T.ffill(inplace= True, axis=0)
        flight_T.bfill(inplace= True, axis=0)
        flight = flight_T.T
        flight = torch.tensor(flight.to_numpy(), dtype=torch.float32)        
        flight = flight.unsqueeze(dim=0)

        return flight, flight_id



        

