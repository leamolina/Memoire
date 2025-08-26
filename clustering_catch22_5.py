# -*- coding: utf-8 -*-
"""
Clustering feature-based des séries temporelles (version optimisée)
- Features "Business Hours" (moyenne par heure x jour -> 7x24 = 168)
- Moyenne par heure (24) et par jour (7) sur toute la semaine
- Statistiques hebdomadaires (min/mean/max par semaine + agrégats inter-semaines)
- tsfresh (features automatiques) avec profil configurable (minimal / efficient / lite)
- Filtrage variance & corrélation
- Normalisation + PCA (retenir 90% variance) + KMeans (k=2..10) + sélection par silhouette
- Visualisations : PCA 2D + heatmaps 7x24 par cluster

Prérequis :
    pip install pandas numpy matplotlib seaborn scikit-learn tsfresh tqdm
"""

import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.cluster import KMeans

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import (
    EfficientFCParameters,
    MinimalFCParameters,
)

# --- configuration globale ---
RANDOM_STATE = 42
RESULTS_DIR = "results_catch22_5"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Contrôles
USE_TSFRESH = True                      # Active l'extraction tsfresh
TSFRESH_PROFILE = "lite"                # "minimal", "efficient", ou "lite" (sélectif recommandé)
VAR_THRESHOLD = 1e-3                    # Seuil de variance minimale
CORR_THRESHOLD = 0.95                   # Seuil de corrélation pour drop features
K_GRID = list(range(2, 11))             # k à tester pour KMeans

# --- PCA pour le MODELE (pas que la visu) ---
USE_PCA_FOR_MODEL = True
PCA_VARIANCE_TO_KEEP = 0.90             # Conserver 90% de variance
# --- PCA 2D pour la visualisation (inchangé) ---
PCA_COMPONENTS_FOR_PLOT = 2

# Parallélisme
N_JOBS_TSFRESH = -1                     # Utiliser tous les coeurs pour tsfresh


# ---------------------------------------------------------------------
# tsfresh param helpers
# ---------------------------------------------------------------------
def get_tsfresh_parameters(profile: str = "lite"):
    profile = profile.lower().strip()
    if profile == "minimal":
        return MinimalFCParameters()
    if profile == "efficient":
        return EfficientFCParameters()

    # Profil "lite" corrigé
    return {
        "agg_linear_trend": [{"attr": "slope", "chunk_len": 10, "f_agg": "mean"}],
        "autocorrelation": [{"lag": l} for l in [1, 2, 24, 168] if l > 0],
        # ⚠️ Corrigé: on fournit aussi segment_focus (au moins un par num_segments)
        "energy_ratio_by_chunks": [
            {"num_segments": n, "segment_focus": s}
            for n in [5, 10]
            for s in [0, n - 1]   # prise des segments aux bords (rapide et informatif)
        ],
        "fft_aggregated": [{"aggtype": agg} for agg in ["centroid", "variance"]],
        "variation_coefficient": None,
        "quantile": [{"q": q} for q in [0.1, 0.5, 0.9]],
        "mean": None,
        "median": None,
        "standard_deviation": None,
        "maximum": None,
        "minimum": None,
        "absolute_sum_of_changes": None,
        "cid_ce": [{"normalize": True}],
        "count_above_mean": None,
        "count_below_mean": None,
    }


# ---------------------------------------------------------------------
# Utilitaires de features
# ---------------------------------------------------------------------
def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col])
    return df


def build_business_hours_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Construit :
      - bh_7x24 : moyenne par (day_of_week, hour) -> 168 features
      - hourly_24 : moyenne par heure (24) agrégée sur la semaine
      - daily_7 : moyenne par jour (7) agrégée sur la semaine
    """
    tmp = df.copy()
    tmp["day_of_week"] = tmp["event_date_time"].dt.dayofweek  # 0=lundi … 6=dimanche
    tmp["hour"] = tmp["event_date_time"].dt.hour

    # 7x24 (colonnes multiindex -> flatten)
    bh = (
        tmp.groupby(["ci_name", "day_of_week", "hour"])["event_value"]
        .mean()
        .unstack(["day_of_week", "hour"])
        .fillna(0.0)
    )
    bh.columns = [f"BH_d{d}_h{h}" for d, h in bh.columns]

    # Moyenne par heure (24)
    hourly_24 = (
        tmp.groupby(["ci_name", "hour"])["event_value"]
        .mean()
        .unstack("hour")
        .fillna(0.0)
    )
    hourly_24.columns = [f"HR_mean_h{h}" for h in hourly_24.columns]

    # Moyenne par jour (7)
    daily_7 = (
        tmp.groupby(["ci_name", "day_of_week"])["event_value"]
        .mean()
        .unstack("day_of_week")
        .fillna(0.0)
    )
    daily_7.columns = [f"DOW_mean_d{d}" for d in daily_7.columns]

    return bh, hourly_24, daily_7


def build_weekly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule min/mean/max par semaine, puis agrège inter-semaines pour chaque ci_name :
      - mean_of_week_mean, std_of_week_mean
      - mean_of_week_min,  min_of_week_min
      - mean_of_week_max,  max_of_week_max
    Et quelques stats globales.
    """
    tmp = df.copy()
    isocal = tmp["event_date_time"].dt.isocalendar()
    tmp["iso_year"] = isocal.year.astype(int)
    tmp["iso_week"] = isocal.week.astype(int)

    wk = (
        tmp.groupby(["ci_name", "iso_year", "iso_week"])["event_value"]
        .agg(week_mean="mean", week_min="min", week_max="max")
        .reset_index()
    )

    agg = (
        wk.groupby("ci_name")
        .agg(
            mean_of_week_mean=("week_mean", "mean"),
            std_of_week_mean=("week_mean", "std"),
            mean_of_week_min=("week_min", "mean"),
            min_of_week_min=("week_min", "min"),
            mean_of_week_max=("week_max", "mean"),
            max_of_week_max=("week_max", "max"),
        )
        .fillna(0.0)
    )

    # Stats globales de la série
    global_stats = (
        tmp.groupby("ci_name")["event_value"]
        .agg(global_mean="mean", global_std="std", global_min="min", global_max="max")
        .fillna(0.0)
    )

    return agg.join(global_stats, how="outer").fillna(0.0)


def build_tsfresh_features(df: pd.DataFrame, profile: str = TSFRESH_PROFILE) -> pd.DataFrame:
    """
    tsfresh sur (ci_name, event_date_time, event_value).
    Profil configurable + imputation auto.
    """
    if not USE_TSFRESH:
        return pd.DataFrame()

    ts_df = df.loc[:, ["ci_name", "event_date_time", "event_value"]].copy()
    ts_df = ts_df.sort_values(["ci_name", "event_date_time"])

    fc_parameters = get_tsfresh_parameters(profile)

    features = extract_features(
        ts_df,
        column_id="ci_name",
        column_sort="event_date_time",
        column_value="event_value",
        default_fc_parameters=fc_parameters,
        disable_progressbar=False,
    )
    features = impute(features)
    return features  # index = ci_name


# ---------------------------------------------------------------------
# Préparation features & filtrage
# ---------------------------------------------------------------------
def variance_and_corr_filter(
    X: pd.DataFrame,
    var_threshold: float = VAR_THRESHOLD,
    corr_threshold: float = CORR_THRESHOLD,
) -> pd.DataFrame:
    """
    1) retire les features trop peu variables (VarianceThreshold)
    2) retire les features très corrélées (> corr_threshold) en gardant la première
    """
    if X.empty:
        return X

    # 1. Variance
    vt = VarianceThreshold(threshold=var_threshold)
    X_v = pd.DataFrame(vt.fit_transform(X), index=X.index, columns=X.columns[vt.get_support()])

    # 2. Corrélation
    corr = X_v.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    X_f = X_v.drop(columns=to_drop)
    return X_f


# ---------------------------------------------------------------------
# Clustering & évaluation
# ---------------------------------------------------------------------
def evaluate_labels(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if len(np.unique(labels)) <= 1:
        return dict(silhouette=-1.0, calinski=-1.0, davies=-1.0)
    try:
        sil = silhouette_score(X, labels)
    except Exception:
        sil = -1.0
    try:
        cal = calinski_harabasz_score(X, labels)
    except Exception:
        cal = -1.0
    try:
        dav = davies_bouldin_score(X, labels)
    except Exception:
        dav = -1.0
    return dict(silhouette=sil, calinski=cal, davies=dav)


def run_kmeans_grid(X: np.ndarray, k_grid: List[int]) -> pd.DataFrame:
    rows = []
    for k in k_grid:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        mets = evaluate_labels(X, labels)
        rows.append(dict(algorithm="KMeans", k=k, **mets, labels=labels))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------
def plot_pca_scatter(X_scaled: np.ndarray, labels: np.ndarray, outpath: str, title: str):
    pca = PCA(n_components=PCA_COMPONENTS_FOR_PLOT, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X2[:, 0], X2[:, 1], c=labels, alpha=0.8, cmap="viridis")
    plt.colorbar(sc, label="Cluster")
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_cluster_business_hour_heatmaps(
    df: pd.DataFrame, labels: np.ndarray, out_dir: str
):
    """
    Pour chaque cluster, calcule la matrice moyenne 7x24 et affiche une heatmap.
    """
    tmp = df.copy()
    tmp["day_of_week"] = tmp["event_date_time"].dt.dayofweek
    tmp["hour"] = tmp["event_date_time"].dt.hour

    assign = pd.DataFrame({"ci_name": df["ci_name"].drop_duplicates().values, "cluster": labels})
    tmp = tmp.merge(assign, on="ci_name", how="left")

    for cl in sorted(assign["cluster"].unique()):
        sub = tmp[tmp["cluster"] == cl]
        if sub.empty:
            continue
        mat = (
            sub.groupby(["day_of_week", "hour"])["event_value"]
            .mean()
            .unstack("hour")
            .reindex(index=range(7), columns=range(24), fill_value=np.nan)
        )
        plt.figure(figsize=(12, 4.5))
        sns.heatmap(mat, annot=False, cmap="viridis")
        plt.title(f"Cluster {cl} – Moyenne event_value par jour (0=Lun) et heure")
        plt.xlabel("Heure")
        plt.ylabel("Jour de la semaine")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cluster_{cl}_heatmap_7x24.png"), dpi=300)
        plt.close()


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")

    # 1) Charger les données
    data_path = "Data_night/data_complete_interface.csv"
    print(f"Chargement des données depuis {data_path} ...")
    df = pd.read_csv(data_path)
    assert {"ci_name", "event_date_time", "event_value"}.issubset(df.columns), \
        "Colonnes requises manquantes: ci_name, event_date_time, event_value"

    df = ensure_datetime(df, "event_date_time")
    df = df.sort_values(["ci_name", "event_date_time"])
    print(f"Nb équipements: {df['ci_name'].nunique()}, nb lignes: {len(df)}")

    # 2) Features Business Hours + agrégats heure/jour
    print("Construction des features Business Hours (7x24), heure (24), jour (7)...")
    bh_7x24, hourly_24, daily_7 = build_business_hours_features(df)

    # 3) Stats hebdo
    print("Construction des statistiques hebdomadaires...")
    weekly_stats = build_weekly_stats(df)

    # 4) tsfresh (profil limité + parallélisme)
    if USE_TSFRESH:
        print(f"Extraction des features tsfresh (profil: {TSFRESH_PROFILE}) ...")
        tsf = build_tsfresh_features(df, profile=TSFRESH_PROFILE)
        print(f"tsfresh shape: {tsf.shape}")
    else:
        tsf = pd.DataFrame()

    # 5) Fusion de toutes les features
    print("Fusion de toutes les features...")
    features = bh_7x24.join([hourly_24, daily_7, weekly_stats], how="outer")
    if not tsf.empty:
        features = features.join(tsf, how="outer")

    # Nettoyage NaN -> 0
    features = features.fillna(0.0)
    features.to_csv(os.path.join(RESULTS_DIR, "all_features_raw.csv"))
    print(f"Features fusionnées: {features.shape[1]} colonnes")

    # 6) Filtrage variance & corrélation
    print("Filtrage (variance + corrélation)...")
    features_f = variance_and_corr_filter(features)
    print(f"Features après filtrage: {features_f.shape[1]} colonnes")
    features_f.to_csv(os.path.join(RESULTS_DIR, "all_features_filtered.csv"))

    # 7) Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_f.values)

    # 8) PCA pour le MODELE (réduction bruit + vitesse + métriques)
    if USE_PCA_FOR_MODEL:
        print(f"PCA pour le modèle (conserver {int(PCA_VARIANCE_TO_KEEP*100)}% variance)...")
        pca_model = PCA(n_components=PCA_VARIANCE_TO_KEEP, random_state=RANDOM_STATE)
        X_model = pca_model.fit_transform(X_scaled)
        print(f"Dimensions après PCA (modèle): {X_model.shape}")
        # Sauvegarde variance expliquée cumulée
        np.savetxt(os.path.join(RESULTS_DIR, "pca_explained_variance_ratio.txt"),
                   pca_model.explained_variance_ratio_)
    else:
        X_model = X_scaled

    # 9) Clustering KMeans (k=2..10) sur l'espace réduit
    print("Clustering KMeans sur grille de k ...")
    grid_df = run_kmeans_grid(X_model, K_GRID)

    # Déballer labels pour sauvegarde
    labels_map = {}
    for _, row in grid_df.iterrows():
        labels_map[(row["algorithm"], int(row["k"]))] = row["labels"]

    # Retirer la colonne labels (non sérialisable direct)
    grid_out = grid_df.drop(columns=["labels"])
    grid_out.to_csv(os.path.join(RESULTS_DIR, "clustering_metrics.csv"), index=False)
    print("Métriques sauvegardées -> clustering_metrics.csv")

    # 10) Choix de la meilleure config par silhouette (avec garde-fou)
    grid_sorted = grid_out.sort_values(["silhouette", "calinski", "davies"], ascending=[False, False, True])
    best_row = grid_sorted.iloc[0]
    best_k = int(best_row["k"])
    best_labels = labels_map[("KMeans", best_k)]

    # 11) Sauvegarde assignations
    assign = pd.DataFrame({
        "ci_name": features_f.index,
        "cluster": best_labels
    })
    assign.to_csv(os.path.join(RESULTS_DIR, f"cluster_assignments_k{best_k}.csv"), index=False)
    print(f"Meilleur k = {best_k} | silhouette={best_row['silhouette']:.4f}, "
          f"calinski={best_row['calinski']:.1f}, davies={best_row['davies']:.3f}")

    # 12) Visualisations (PCA 2D sur les données standardisées d'origine)
    print("Visualisations PCA + heatmaps 7x24...")
    plot_pca_scatter(
        X_scaled,
        best_labels,
        os.path.join(RESULTS_DIR, f"pca_scatter_k{best_k}.png"),
        f"PCA 2D – KMeans (k={best_k})"
    )

    # Heatmaps business hours par cluster
    plot_cluster_business_hour_heatmaps(
        df=df[["ci_name", "event_date_time", "event_value"]].copy(),
        labels=best_labels,
        out_dir=RESULTS_DIR
    )

    # 13) Récap
    recap = {
        "best_k": best_k,
        "silhouette": round(float(best_row["silhouette"]), 6),
        "calinski_harabasz": round(float(best_row["calinski"]), 3),
        "davies_bouldin": round(float(best_row["davies"]), 6),
        "n_features_raw": int(features.shape[1]),
        "n_features_filtered": int(features_f.shape[1]),
        "n_equipements": int(df["ci_name"].nunique()),
        "n_rows": int(len(df)),
        "pca_model_used": int(USE_PCA_FOR_MODEL),
        "tsfresh_profile": TSFRESH_PROFILE,
    }
    pd.Series(recap).to_csv(os.path.join(RESULTS_DIR, "recap.csv"))
    print("Terminé. Résultats dans:", RESULTS_DIR)


if __name__ == "__main__":
    main()
