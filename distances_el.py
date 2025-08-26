import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
from tqdm import tqdm
import os
from joblib import Parallel, delayed
from datetime import datetime

import inspect
from aeon.distances import (
    dtw_distance, 
    ddtw_distance,
    wdtw_distance,
    wddtw_distance,
    lcss_distance,
    erp_distance,
    edr_distance,
    msm_distance,
    twe_distance
)

# Dictionnaire des fonctions de distance
DISTANCE_FUNCTIONS = {
    'dtw': dtw_distance,
    'ddtw': ddtw_distance,
    'wdtw': wdtw_distance,
    'wddtw': wddtw_distance,
    'lcss': lcss_distance,
    'erp': erp_distance,
    'edr': edr_distance,
    'msm': msm_distance,
    'twe': twe_distance
}


warnings.filterwarnings('ignore')

# Créer un dossier pour les résultats
results_dir = f"clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(results_dir, exist_ok=True)


def filter_data(df):
    """
    Filtre le DataFrame pour ne garder que les CI "normaux" et tronque/padde chaque série à la longueur médiane.
    Le padding est fait avec la moyenne de event_value de la série.

    Paramètres
    ----------
    df : pd.DataFrame
        Doit contenir au moins les colonnes :
        - 'ci_name'
        - 'event_date_time' (datetime ou convertible)
        - 'event_value'

    Retourne
    -------
    df_filtered : pd.DataFrame
        DataFrame trié, filtré et mis à longueur fixe (L lignes par ci_name).
    keep_names : list[str]
        Liste des ci_name conservés.
    L : int
        Longueur à laquelle chaque série a été normalisée.
    """
    df = df.copy()
    # 1) Conversion en datetime si nécessaire
    if not pd.api.types.is_datetime64_any_dtype(df['event_date_time']):
        df['event_date_time'] = pd.to_datetime(df['event_date_time'])

    # 2) Calcul des longueurs
    lengths = df.groupby('ci_name').size()

    # 3) Calcul des bornes IQR
    Q1, Q3 = lengths.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lf, uf = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    # 4) Longueur cible = médiane
    L = int(lengths.median())

    # 5) Sélection des ci_name "normaux"
    keep_names = lengths[(lengths >= lf) & (lengths <= uf)].index.tolist()

    # 6) Tri global
    df_sorted = df[df['ci_name'].isin(keep_names)] \
                  .sort_values(['ci_name', 'event_date_time'])

    # 7) Construction du df filtré avec padding (à la moyenne) / troncature
    rows = []
    cols = df_sorted.columns
    for name in keep_names:
        grp = df_sorted[df_sorted['ci_name'] == name]
        n = len(grp)
        if n >= L:
            sub = grp.head(L)
        else:
            # padding
            pad_n = L - n
            mean_val = grp['event_value'].mean()
            pad_dict = {}
            for col in cols:
                if col == 'ci_name':
                    pad_dict[col] = [name] * pad_n
                elif col == 'event_date_time':
                    pad_dict[col] = [pd.NaT] * pad_n
                elif col == 'event_value':
                    pad_dict[col] = [mean_val] * pad_n
                else:
                    # si d'autres colonnes existent, on peut padder par NaN
                    pad_dict[col] = [np.nan] * pad_n
            pad_df = pd.DataFrame(pad_dict, columns=cols)
            sub = pd.concat([grp, pad_df], ignore_index=True)
        rows.append(sub)

    df_filtered = pd.concat(rows, ignore_index=True)
    return df_filtered, keep_names, L

# Fonction pour charger et préparer les données
def prepare_data(df):

    # On ne garde que les colonnes importantes pour le clustering de séries temporelles
    df = df[['ci_name', 'event_value', 'event_date_time']]
    df, keep_names, L = filter_data(df)
    # Convertir event_date_time en datetime si ce n'est pas déjà fait
    if not pd.api.types.is_datetime64_any_dtype(df['event_date_time']):
        df['event_date_time'] = pd.to_datetime(df['event_date_time'])
    
    # Trier par ci_name et event_date_time
    df = df.sort_values(['ci_name', 'event_date_time'])
    
    # Normalisation des valeurs entre 0 et 1
    # Même si les données sont déjà entre 0 et 100, la normalisation est recommandée
    # pour les distances élastiques qui sont sensibles à l'échelle
    scaler = MinMaxScaler()
    df['event_value_normalized'] = scaler.fit_transform(df[['event_value']])
    
    # Créer un dictionnaire de séries temporelles par ci_name
    series_dict = {}
    for name, group in df.groupby('ci_name'):
        series_dict[name] = group['event_value_normalized'].values
    
    # Convertir en tableau numpy pour tslearn
    # Nous devons avoir des séries de même longueur
    # Trouver la longueur minimale des séries
    min_length = min(len(series) for series in series_dict.values())
    
    # Tronquer toutes les séries à cette longueur minimale
    series_list = [series[:min_length] for series in series_dict.values()]
    ci_names = list(series_dict.keys())
    for i, series in enumerate(series_list):
        if np.isnan(series).any():
            print(f"Warning: Series {i} ({ci_names[i]}) contains NaN values")
        
        if np.isinf(series).any():
            print(f"Warning: Series {i} ({ci_names[i]}) contains infinite values")
        
        if len(np.unique(series)) == 1:
            print(f"Warning: Series {i} ({ci_names[i]}) is constant (value: {series[0]})")
        
        if len(series) < 10:  # Seuil arbitraire pour une série très courte
            print(f"Warning: Series {i} ({ci_names[i]}) is very short (length: {len(series)})")
    # Convertir en tableau 3D pour tslearn (n_series, n_timestamps, n_features)
    X = np.array(series_list).reshape(len(series_list), min_length, 1)
    
    return X, ci_names, min_length


# Fonction pour calculer une paire de distance (pour parallélisation)
def compute_distance_pair(i, j, X, distance_type, **kwargs):
    """Calcule la distance entre les séries temporelles i et j."""
    # Extraire les séries temporelles
    ts1 = X[i].ravel() if X[i].ndim > 1 else X[i]
    ts2 = X[j].ravel() if X[j].ndim > 1 else X[j]
    
    try:
        # Obtenir la fonction de distance appropriée
        distance_func = DISTANCE_FUNCTIONS.get(distance_type)
        
        if distance_func is None:
            raise ValueError(f"Distance type {distance_type} not supported")
        
        # Calculer la distance
        dist = distance_func(ts1, ts2, **kwargs)
        
        return dist
    except Exception as e:
        warnings.warn(f"Error computing {distance_type} distance between series {i} and {j}: {str(e)}")
        return float('inf')  # Valeur par défaut en cas d'erreur

# Fonction principale pour calculer la matrice de distance
def compute_distance_matrix(X, distance_type, n_jobs=1, **kwargs):
    """
    Calcule la matrice de distance entre toutes les paires de séries temporelles.
    
    Parameters:
    -----------
    X : array-like, shape (n_series, n_timestamps) ou (n_series, n_timestamps, n_features)
        Les séries temporelles.
    distance_type : str
        Type de distance à calculer ('dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp', 'msm', 'twe').
    n_jobs : int, default=1
        Nombre de jobs parallèles. -1 pour utiliser tous les processeurs.
    **kwargs : dict
        Paramètres supplémentaires pour la fonction de distance.
        
    Returns:
    --------
    dist_matrix : array-like, shape (n_series, n_series)
        Matrice de distance.
    """
    # Vérifier la forme de X et la convertir si nécessaire
    if X.ndim == 3 and X.shape[2] == 1:
        X = X[:, :, 0]  # Convertir (n_series, n_timestamps, 1) en (n_series, n_timestamps)
    
    n_series = X.shape[0]
    dist_matrix = np.zeros((n_series, n_series))
    
    # Générer toutes les paires (i,j) où i <= j
    pairs = [(i, j) for i in range(n_series) for j in range(i, n_series)]
    
    if n_jobs == 1:
        # Version séquentielle avec barre de progression
        for i, j in tqdm(pairs, desc=f"Computing {distance_type} distances"):
            dist = compute_distance_pair(i, j, X, distance_type, **kwargs)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    else:
        # Version parallèle
        print(f"Computing {distance_type} distances in parallel with {n_jobs} jobs...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_distance_pair)(i, j, X, distance_type, **kwargs) 
            for i, j in tqdm(pairs, desc="Preparing jobs")
        )
        
        # Remplir la matrice
        for (i, j), dist in zip(pairs, results):
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    # Vérifier si la matrice contient des valeurs infinies
    # if np.isinf(dist_matrix).any():
    #     warnings.warn(f"Distance matrix contains infinite values. Check your data or parameters.")

    # Option 2 : 
    # Après avoir calculé la matrice de distance
    if np.isinf(dist_matrix).any():
        warnings.warn(f"Distance matrix contains infinite values. Replacing with max finite value.")
        max_finite = np.max(dist_matrix[np.isfinite(dist_matrix)])
        dist_matrix[np.isinf(dist_matrix)] = max_finite * 2  # Double du max comme valeur de remplacement

    
    return dist_matrix

# Fonction pour évaluer les clusters
# Fonction pour évaluer les clusters
def evaluate_clusters(X, distance_matrix, n_clusters_range):
    results = []
    
    for n_clusters in tqdm(n_clusters_range, desc="Evaluating clusters"):
        try:
            # Appliquer K-medoids
            kmedoids = KMedoids(
                n_clusters=n_clusters,
                metric="precomputed",
                init="k-medoids++",
                max_iter=300,
                random_state=42
            )
            
            cluster_labels = kmedoids.fit_predict(distance_matrix)
            
            # Calculer les métriques d'évaluation
            # Note: certaines métriques comme ARI, NMI nécessitent des labels vrais
            # que nous n'avons pas, donc nous utilisons des métriques internes
            
            # Convertir X en 2D pour silhouette_score
            X_2d = X.reshape(X.shape[0], -1)
            
            silhouette = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
            davies_bouldin = davies_bouldin_score(X_2d, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(X_2d, cluster_labels)
            
            # Stocker les résultats
            results.append({
                'n_clusters': n_clusters,
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'calinski_harabasz': calinski_harabasz,
                'labels': cluster_labels
            })
        except Exception as e:
            print(f"Error with {n_clusters} clusters: {e}")
    
    return results

# Fonction pour visualiser les résultats
def visualize_results(results, distance_type, results_dir):
    # Créer un DataFrame avec les résultats
    df_results = pd.DataFrame([
        {
            'n_clusters': r['n_clusters'],
            'silhouette': r['silhouette'],
            'davies_bouldin': r['davies_bouldin'],
            'calinski_harabasz': r['calinski_harabasz']
        }
        for r in results
    ])
    
    # Sauvegarder les résultats dans un CSV
    df_results.to_csv(f"{results_dir}/{distance_type}_metrics.csv", index=False)
    
    # Créer des graphiques pour chaque métrique
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(df_results['n_clusters'], df_results['silhouette'], 'o-', label='Silhouette')
    plt.title(f'Silhouette Score - {distance_type}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(df_results['n_clusters'], df_results['davies_bouldin'], 'o-', label='Davies-Bouldin')
    plt.title(f'Davies-Bouldin Index - {distance_type}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(df_results['n_clusters'], df_results['calinski_harabasz'], 'o-', label='Calinski-Harabasz')
    plt.title(f'Calinski-Harabasz Index - {distance_type}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{distance_type}_metrics.png")
    plt.close()
    
    return df_results

# Fonction pour visualiser les clusters
def visualize_clusters(X, best_labels, ci_names, distance_type, n_clusters, results_dir):
    plt.figure(figsize=(15, 10))
    
    # Créer un dictionnaire pour stocker les séries par cluster
    cluster_series = {i: [] for i in range(n_clusters)}
    cluster_names = {i: [] for i in range(n_clusters)}
    
    for i, label in enumerate(best_labels):
        cluster_series[label].append(X[i, :, 0])
        cluster_names[label].append(ci_names[i])
    
    # Tracer les séries temporelles par cluster
    for cluster_id, series_list in cluster_series.items():
        plt.subplot(n_clusters, 1, cluster_id + 1)
        for series in series_list:
            plt.plot(series, 'k-', alpha=0.2)
        
        # Calculer et tracer la moyenne du cluster
        if series_list:
            mean_series = np.mean(series_list, axis=0)
            plt.plot(mean_series, 'r-', linewidth=2)
        
        plt.title(f'Cluster {cluster_id+1} (n={len(series_list)})')
        plt.ylim(0, 1)
        
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{distance_type}_clusters_{n_clusters}.png")
    plt.close()
    
    # Sauvegarder les noms des CI par cluster
    cluster_df = pd.DataFrame({
        'ci_name': ci_names,
        'cluster': best_labels
    })
    cluster_df.to_csv(f"{results_dir}/{distance_type}_cluster_assignments_{n_clusters}.csv", index=False)
    
    return cluster_df

# Fonction principale
def run_clustering_analysis(df, dataset_name):
    print(f"Analyzing {dataset_name}...")
    
    # Créer un sous-dossier spécifique pour ce dataset
    results_dir = "Analyse_kmedoids"
    basename = os.path.splitext(os.path.basename(dataset_name))[0]
    dataset_dir = os.path.join(results_dir, basename)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Préparer les données
    X, ci_names, min_length = prepare_data(df)
    print(f"Prepared {len(ci_names)} time series with {min_length} points each")
    
    
    radius = min_length // 10
    window_ratio = radius / min_length  # typiquement = 0.1

    distance_params = {
        'dtw':  {'window': window_ratio, 'itakura_max_slope': None},
        'ddtw': {'window': window_ratio, 'itakura_max_slope': None},
        'wdtw':  {'window': window_ratio, 'g': 0.05, 'itakura_max_slope': None},
        'wddtw': {'window': window_ratio, 'g': 0.05, 'itakura_max_slope': None},
        'lcss': {'window': window_ratio, 'epsilon': 0.1, 'itakura_max_slope': None},
        'edr':  {'window': window_ratio, 'epsilon': 0.1, 'itakura_max_slope': None},
        'erp':  {'window': window_ratio, 'g': 0.0, 'g_arr': None, 'itakura_max_slope': None},
        'msm':  {'window': window_ratio, 'independent': True, 'c': 1.0, 'itakura_max_slope': None},
        'twe':  {'window': window_ratio, 'nu': 0.001, 'lmbda': 1.0, 'itakura_max_slope': None}
    }
    # Plage de clusters à tester
    n_clusters_range = range(2, 7)
    
    # Stocker les résultats globaux
    all_results = {}
    best_metrics = {}
    
    # Tester chaque distance
    for distance_type in distance_params.keys():
        print(f"\nTesting {distance_type} distance...")
        
        # Calculer la matrice de distance
        distance_matrix = compute_distance_matrix(X, distance_type, n_jobs=-1, **distance_params[distance_type])
        
        # Évaluer les clusters
        results = evaluate_clusters(X, distance_matrix, n_clusters_range)
        all_results[distance_type] = results
        
        # Visualiser les résultats pour cette distance
        df_results = visualize_results(results, distance_type, dataset_dir)
        
        # Déterminer le nombre optimal de clusters pour cette distance
        # Nous utilisons une combinaison des métriques:
        # - Maximiser Silhouette et Calinski-Harabasz
        # - Minimiser Davies-Bouldin
        
        # Normaliser les métriques pour pouvoir les combiner
        df_results['silhouette_norm'] = (df_results['silhouette'] - df_results['silhouette'].min()) / \
                                        (df_results['silhouette'].max() - df_results['silhouette'].min() + 1e-10)
        
        df_results['davies_bouldin_norm'] = 1 - (df_results['davies_bouldin'] - df_results['davies_bouldin'].min()) / \
                                           (df_results['davies_bouldin'].max() - df_results['davies_bouldin'].min() + 1e-10)
        
        df_results['calinski_harabasz_norm'] = (df_results['calinski_harabasz'] - df_results['calinski_harabasz'].min()) / \
                                              (df_results['calinski_harabasz'].max() - df_results['calinski_harabasz'].min() + 1e-10)
        
        # Score combiné (moyenne des métriques normalisées)
        df_results['combined_score'] = (df_results['silhouette_norm'] + 
                                       df_results['davies_bouldin_norm'] + 
                                       df_results['calinski_harabasz_norm']) / 3
        
        # Trouver le meilleur nombre de clusters
        best_idx = df_results['combined_score'].idxmax()
        best_n_clusters = df_results.loc[best_idx, 'n_clusters']
        best_score = df_results.loc[best_idx, 'combined_score']
        
        print(f"Best number of clusters for {distance_type}: {best_n_clusters} (score: {best_score:.4f})")
        
        # Récupérer les labels pour le meilleur nombre de clusters
        best_result = next(r for r in results if r['n_clusters'] == best_n_clusters)
        best_labels = best_result['labels']
        
        # Visualiser les clusters pour la meilleure configuration
        visualize_clusters(X, best_labels, ci_names, distance_type, best_n_clusters, dataset_dir)
        
        # Stocker les métriques pour comparer les distances
        best_metrics[distance_type] = {
            'n_clusters': best_n_clusters,
            'silhouette': best_result['silhouette'],
            'davies_bouldin': best_result['davies_bouldin'],
            'calinski_harabasz': best_result['calinski_harabasz'],
            'combined_score': best_score
        }
    
    # Comparer les différentes distances et trouver la meilleure
    best_metrics_df = pd.DataFrame.from_dict(best_metrics, orient='index')
    best_metrics_df.to_csv(f"{dataset_dir}/best_metrics_comparison.csv")
    
    # Trouver la meilleure distance basée sur le score combiné
    best_distance = best_metrics_df['combined_score'].idxmax()
    best_n_clusters_overall = best_metrics_df.loc[best_distance, 'n_clusters']
    
    print(f"\n=== RESULTS FOR {dataset_name} ===")
    print(f"Best distance metric: {best_distance}")
    print(f"Optimal number of clusters: {best_n_clusters_overall}")
    print(f"Silhouette score: {best_metrics_df.loc[best_distance, 'silhouette']:.4f}")
    print(f"Davies-Bouldin index: {best_metrics_df.loc[best_distance, 'davies_bouldin']:.4f}")
    print(f"Calinski-Harabasz index: {best_metrics_df.loc[best_distance, 'calinski_harabasz']:.4f}")
    
    # Créer un graphique comparatif des distances
    plt.figure(figsize=(12, 10))
    
    metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'combined_score']
    titles = ['Silhouette Score (higher is better)', 
              'Davies-Bouldin Index (lower is better)', 
              'Calinski-Harabasz Index (higher is better)',
              'Combined Score (higher is better)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(2, 2, i+1)
        
        # Pour Davies-Bouldin, plus bas est meilleur, donc inverser pour la visualisation
        values = best_metrics_df[metric]
        if metric == 'davies_bouldin':
            values = -values
            
        bars = plt.bar(best_metrics_df.index, values)
        
        # Mettre en évidence la meilleure distance
        if metric == 'combined_score':
            highlight_idx = best_metrics_df.index.get_loc(best_distance)
            bars[highlight_idx].set_color('red')
        
        plt.title(title)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{dataset_dir}/distance_comparison.png")
    plt.close()
    
    return best_distance, best_n_clusters_overall, best_metrics_df

def main():
    dataset_name = "Data/data_complete_interface.csv"
    df = pd.read_csv(dataset_name)

    best_distance, best_n_clusters_overall, best_metrics_df = run_clustering_analysis(df, dataset_name)
    print("Best distance : ", best_distance)
    print("Best number of clusters", best_n_clusters_overall)

    print("Best metrics df :")
    print(best_metrics_df.head())

if __name__ == "__main__":
    main()
