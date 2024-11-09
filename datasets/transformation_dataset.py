import torch
import pandas as pd
from torch.utils.data import Dataset

class TransformationDataset(Dataset):
    def __init__(self, flight_id_topath, transformation_method):
        self.flight_id_topath = flight_id_topath.reset_index()
        self.transformation = transformation_method
    
    def __len__(self):
        return len(self.flight_id_topath)
    
    def __getitem__(self, index):
        path = self.flight_data.loc[index, "file_path"]
        flight = pd.read_csv(path, na_values=[' NaN', 'NaN', 'NaN '])
        flight = torch.tensor(flight.to_numpy(), dtype=torch.float32)
        flight_transformed = self.transformation(flight)

        pos_pair = (flight.unsqueeze(dim=0), flight_transformed.unsqueeze(dim=0))
        return pos_pair