from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import tqdm
from torch.cuda.amp import autocast
import torch
import numpy as np
import pandas as pd
import seaborn as sns

SS_PATH = "/mnt/crucial/data/ngafid/exports/loci_dataset_fixed_keys/flight_safety_scores.csv"
AT_PATH = "/mnt/crucial/data/ngafid/exports/loci_dataset_fixed_keys/aircraft_types.csv"


def visualize(model, args, dataloader, dr_methods = ["PCA", "tSNE"]):
    # X:                flight features                 (dataset size, feature size)
    # safety_labels:    the safety score of each flight (dataset size, 1)
    # aircraft_labels:  the aircraft type of each flight (dataset size, 1)
    
    # get flights
    # get features from flights
    # get flight safety scores
    # get flight aircraft
    # TODO: normalize 


    X = []
    ids = []

    model.remove_projector()
    model.eval()

    with torch.no_grad():  
        for images, flight_ids in dataloader:
            images = images.to(args.device)
            with autocast(enabled=args.fp16_precision):
                batch_features = model(images)  
            X.append(batch_features.detach().cpu())
            ids.append(flight_ids)

    X = torch.cat(X, dim=0).numpy()
    ids = torch.cat(ids, dim=0).numpy()

    flight_data = pd.DataFrame({
        'embedding': list(X),
        'flight_id': ids
    })
    safety_scores = pd.read_csv(SS_PATH)
    aircraft_types = pd.read_csv(AT_PATH)
    flight_data = flight_data.merge(safety_scores, on='flight_id', how='inner').merge(aircraft_types, on='flight_id', how='inner')

    # form numpy array of (#flights, embedding_size)
    embeddings = np.stack(flight_data['embedding'].values)
  
    # Round each element to the nearest 0.1
    # rounded_saftey_scores = torch.round(safety_labels * 10) / 10

    if "PCA" in dr_methods:
        normalized_embeddings = StandardScaler().fit_transform(embeddings)
        pca = PCA(n_components=2)
        components = pca.fit_transform(normalized_embeddings)
        flight_data['PC1'] = components[:, 0]
        flight_data['PC2'] = components[:, 1]

        print(
            "explained variance ratio (first two components): %s"
            % str(pca.explained_variance_ratio_)
        )
        graph2(flight_data, 'PC1', 'PC2', 'tfidf', 'PCA')
        graph2(flight_data, 'PC1', 'PC2', 'aircraft_type', 'PCA')
    
    if "TSNE" in dr_methods:
        normalized_embeddings = StandardScaler().fit_transform(embeddings)
        # need to understand TSNE and its arguments better
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        components = tsne.fit_transform(normalized_embeddings)
        flight_data['Dim1'] = components[:, 0]
        flight_data['Dim2'] = components[:, 1]
        graph2(flight_data, 'Dim1', 'Dim2', 'tfidf', 'TSNE')
        graph2(flight_data, 'Dim1', 'Dim2', 'aircraft_type', 'TSNE')


    
    # if "PCA" in dr_methods:
    #     pca = PCA(n_components=2, power_iteration_normalizer='auto')
    #     X_r = pca.fit(features).transform(features)
    #     # Percentage of variance explained for each components
    #     print(
    #         "explained variance ratio (first two components): %s"
    #         % str(pca.explained_variance_ratio_)
    #     )


    #     # graph safety scores
    #     graph(X_r, rounded_saftey_scores, [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "safety scores")
    #     # graph aircraft types
    #     # graph(X_r, aircraft_labels, ["aircraft 1","aircraft 2","aircraft 3",], "aircraft type")
    

    # if "tSNE" in dr_methods:
    #     tsne = TSNE(n_components=2, learning_rate='auto',
    #               init='random', perplexity=3)
    #     X_r = tsne.fit_transform(X)

    #     # graph safety scores
    #     graph(X_r, rounded_saftey_scores, [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "safety scores")
    #     # graph aircraft types
    #     # graph(X_r, aircraft_labels, ["aircraft 1","aircraft 2","aircraft 3",], "aircraft type")
    
def graph2(df, pc1_col, pc2_col, hue_col, dr_type):
    plt.figure(figsize=(10, 8))

    if hue_col == 'tfidf':
        scatter = plt.scatter(
            df[pc1_col],
            df[pc2_col],
            c=df[hue_col],
            cmap='viridis',
            alpha=0.7,
            s=30
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label(hue_col)
    else:
        sns.scatterplot(
            data=df,
            x=pc1_col,
            y=pc2_col,
            hue=hue_col,  
            palette='Set1',
            alpha=0.7,
            s=30
        )
        plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(dr_type + " of SSL Representations")
    plt.xlabel(pc1_col)
    plt.ylabel(pc2_col)
    plt.show()
    
   
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


