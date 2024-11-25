import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset


MAX_ROWS = 7289


class ScorePairDataset(Dataset):
    def __init__(self, pairs, flight_data=None, use_1d: bool=False):
        self.pairs = pairs
        self.flight_data = flight_data
        self.use_1d = use_1d
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        pair = self.pairs["Positive Pairs"][index]     
        path_1 = self.flight_data.loc[int(pair[0]), "file_path"]
        path_2 = self.flight_data.loc[int(pair[1]), "file_path"]

        df_1 = pd.read_csv(path_1, na_values=[' NaN', 'NaN', 'NaN '])
        df_2 = pd.read_csv(path_2, na_values=[' NaN', 'NaN', 'NaN '])
        # df_1 = df_1.applymap(lambda x: x.strip() if isinstance(x, str) else x).drop(columns=df_1.filter(regex='number_events = \d+').columns)
        # df_2 = df_2.applymap(lambda x: x.strip() if isinstance(x, str) else x).drop(columns=df_2.filter(regex='number_events = \d+').columns)


        t_1 = torch.tensor(df_1.to_numpy(), dtype=torch.float32)
        t_2 = torch.tensor(df_2.to_numpy(), dtype=torch.float32)

        # t_1_rows = t_1.size(0)
        # t_2_rows = t_2.size(0)

        # if t_1_rows < MAX_ROWS:
        #     pad_1_rows = MAX_ROWS - t_1_rows
        #
        #     t_1_padded = F.pad(t_1, (0,0,0, pad_1_rows))
        # else:
        #     t_1_padded = t_1
        # 
        #
        # if t_2_rows < MAX_ROWS:
        #     pad_2_rows = MAX_ROWS - t_2_rows
        #
        #     t_2_padded = F.pad(t_2, (0,0,0, pad_2_rows))
        # else:
        #     t_2_padded = t_2
        #

        # breakpoint()

        if self.use_1d:
            flight_pair = (t_1.flatten(), t_2.flatten())
        else:
            flight_pair = (t_1.unsqueeze(dim=0), t_2.unsqueeze(dim=0))

        

        return flight_pair



        

