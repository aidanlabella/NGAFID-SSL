from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import tqdm
from torch.cuda.amp import autocast
import torch
import numpy as np


def visualize(model, args, data, safety_labels, aircraft_labels, dr_metodS = ["PCA", "tSNE"]):
    # X:                flight features                 (dataset size, feature size)
    # safety_labels:    the safety score of each flight (dataset size, 1)
    # aircraft_labels:  the aircraft type of each flight (dataset size, 1)
    
    # get flights
    # get features from flights
    # get flight safety scores
    # get flight aircraft
    # TODO: normalize 
    X = []
    with torch.no_grad():  
        for images in tqdm(data):
            images = images.to(args.device)

            with autocast(enabled=args.fp16_precision):
                batch_features = model(images)  
            X.append(batch_features.detach().cpu())
    X = torch.cat(X, dim=0)

    
    # Round each element to the nearest 0.1
    rounded_saftey_scores = torch.round(safety_labels * 10) / 10
    
    if "PCA" in dr_metods:
        pca = PCA(n_components=2, power_iteration_normalizer='auto')
        X_r = pca.fit(features).transform(features)
        # Percentage of variance explained for each components
        print(
            "explained variance ratio (first two components): %s"
            % str(pca.explained_variance_ratio_)
        )

        # graph safety scores
        graph(X_r, rounded_saftey_scores, [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "safety scores")
        # graph aircraft types
        # graph(X_r, aircraft_labels, ["aircraft 1","aircraft 2","aircraft 3",], "aircraft type")
    

    if "tSNE" in dr_metods:
        tsne = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3)
        X_r = tsne.fit_transform(X)

        # graph safety scores
        graph(X_r, rounded_saftey_scores, [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "safety scores")
        # graph aircraft types
        # graph(X_r, aircraft_labels, ["aircraft 1","aircraft 2","aircraft 3",], "aircraft type")
    
    
    

   

def graph(X_r, y, label_names, label_type):
    plt.figure()
    num_labels = len(label_names)
    colors_options = plt.cm.get_cmap('tab20', num_labels)
    colors = [colors(i) for i in range(num_labels)]
    lw = 2

    for color, i, target_name in zip(colors, torch.arrange(num_labels), label_names):
        plt.scatter(
            X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of NGAFID dataset: " + label_type)
    plt.show()