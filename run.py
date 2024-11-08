import argparse
import torch
import torch.backends.cudnn as cudnn
import pandas as pd

from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from datasets.tf_idf import ScoreDatasetGenerator
from datasets.flight_score_dataset import ScorePairDataset
from sample_flights.combine_flight_data import flight_paths

SS_PATH = "/mnt/crucial/data/ngafid/exports/loci_dataset_fixed_keys/flight_safety_scores.csv"

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
# parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

def dataloader_function(batch):
    first_elements, second_elements = zip(*batch)
    batch_combined = first_elements + second_elements
    batch_combined = torch.stack(batch_combined, dim=0)

    return batch_combined

def get_pos_pairs(non_zero=False):
    scores = pd.read_csv(SS_PATH, index_col='flight_id')
    flights_df = flight_paths()

    if non_zero:
      scores = scores[scores['tfidf'] > 0]

    merged_flights = pd.merge(scores, flights_df, on='flight_id', how='inner')

    data = merged_flights

    data = data.sort_values(by="tfidf").reset_index(drop=False)
    pairs = [(data['flight_id'][i], data['flight_id'][i+1]) for i in range(0,len(data)-1, 2)]
    
    pair_df = pd.DataFrame(
      {
        "Positive Pairs": pairs
      }
    )

    return pair_df    



def main():
    args = parser.parse_args()
    # assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
        args.gpu_index = 1
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    args.device = torch.device('cuda:1')

    # score_generator = ScoreDatasetGenerator()

    all_pairs = get_pos_pairs()
    non_zero_pairs = get_pos_pairs(non_zero=False)
    
    flight_id_to_paths = flight_paths()

    flight = pd.read_csv(flight_id_to_paths['file_path'][951])
    flight = flight.iloc[:, 2:]

    dataset = ScorePairDataset(all_pairs, flight_id_to_paths)

    train_set = dataset

    # train_data_size = int(dataset_size * .7)
    # test_data_size = int(dataset_size * .2)
    # val_data_size = train_data_size - test_data_size
    
    batch_size = 16
    num_workers = 0
    # train_set, test_set, val_set = torch.utils.data.random_split(dataset, [train_data_size, test_data_size, val_data_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=dataloader_function)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)
    
    
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
    
    args.batch_size = batch_size
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)








# def old():
#     args = parser.parse_args()
#     # assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
#     # check if gpu training is available
#     if not args.disable_cuda and torch.cuda.is_available():
#         args.device = torch.device('cuda')
#         cudnn.deterministic = True
#         cudnn.benchmark = True
#     else:
#         args.device = torch.device('cpu')
#         args.gpu_index = -1

#     # TODO: change
    
#     dataset = ContrastiveLearningDataset(args.data)

#     train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle=True,
#         num_workers=args.workers, pin_memory=True, drop_last=True)

#     model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

#     optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
#                                                            last_epoch=-1)

#     #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
#     with torch.cuda.device(args.gpu_index):
#         simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
#         simclr.train(train_loader)


if __name__ == "__main__":
    main()
