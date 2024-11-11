import pandas as pd
import torch
import torch.backends.cudnn as cudnn

from datetime import datetime
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from datasets.tf_idf import ScoreDatasetGenerator
from datasets.flight_score_dataset import ScorePairDataset
from sample_flights.combine_flight_data import flight_paths

SS_PATH = "/mnt/crucial/data/ngafid/exports/loci_dataset_fixed_keys/flight_safety_scores.csv"

def main():
    flight_id_to_paths = flight_paths()
    dataset = ScorePairDataset(all_pairs, flight_id_to_paths)

    train_set = dataset
    device = torch.device('cuda:1')

    # train_data_size = int(dataset_size * .7)
    # test_data_size = int(dataset_size * .2)
    # val_data_size = train_data_size - test_data_size
    
    batch_size = 16
    num_workers = 0
    # train_set, test_set, val_set = torch.utils.data.random_split(dataset, [train_data_size, test_data_size, val_data_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=dataloader_function)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

if __name__ == '__main__':
    main()
