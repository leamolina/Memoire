import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import SGDRegressor
from sklearn.base import clone

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100

# --- Réglages globaux rapides ---
BIG_CLUSTER_ROWS = 1_000_000        # seuil à partir duquel on passe en "mode rapide"
TUNE_MAX_ROWS    = 200_000          # sous-échantillon max pour l'optim des hyperparams
CV_FOLDS_BIG     = 3                # CV réduite pour gros clusters
CV_FOLDS_SMALL   = 5

TRIALS = {                           # nb d'essais Optuna par modèle
    'LinearRegression': 1,           # pas d'espace de recherche
    'Ridge': 25,
    'Lasso': 25,
    'ElasticNet': 40,
    'RandomForest': 30,              # on garde petit
    'GradientBoosting': 40,
    'XGBoost': 60,
    'LightGBM': 60,
    'CatBoost': 60,
    'SVR': 30,
}

# Pruner: stoppe tôt les essais mediocres
import optuna
PRUNER = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)


# Création du dossier pour sauvegarder les résultats
os.makedirs("TimeSeries_forecasting", exist_ok=True)

# 1. Récupération des données
def load_data():
    print("Chargement des données...")

    # --- Données principales
    df = pd.read_csv('Data_night/data_complete_interface.csv')

    # --- Assignations de clusters
    cluster_df = pd.read_csv('results_catch22_6/Interface/cluster_assignments_KMeans_Subspace_spectral_3.csv')

    # Harmonisation des noms de colonnes attendues
    # (ajuste si tes fichiers utilisent d'autres noms)
    if 'ci_name' not in df.columns:
        # essaies courants : 'ci', 'CI_NAME'
        for alt in ['ci', 'CI_NAME', 'device_name']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'ci_name'})
                break

    if 'ci_name' not in cluster_df.columns:
        for alt in ['ci', 'CI_NAME', 'device_name']:
            if alt in cluster_df.columns:
                cluster_df = cluster_df.rename(columns={alt: 'ci_name'})
                break

    if 'cluster' not in cluster_df.columns:
        for alt in ['cluster_id', 'Cluster', 'CLUSTER']:
            if alt in cluster_df.columns:
                cluster_df = cluster_df.rename(columns={alt: 'cluster'})
                break
    """
    # --- Features d'extraction (optionnelles)
    features_df = None
    candidate_paths = [
        'features_df.csv'
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                features_df = pd.read_csv(p)
                # Harmonise ci_name si besoin
                if 'ci_name' not in features_df.columns:
                    for alt in ['ci', 'CI_NAME', 'device_name']:
                        if alt in features_df.columns:
                            features_df = features_df.rename(columns={alt: 'ci_name'})
                            break
                print("Features d'extraction chargées avec succès.")
                break
            except Exception as e:
                print(f"Impossible de charger {p}: {e}")
    """
    # --- Vérifs minimales
    missing_cols = []
    if 'ci_name' not in df.columns: missing_cols.append("df.ci_name")
    if 'ci_name' not in cluster_df.columns: missing_cols.append("cluster_df.ci_name")
    if 'cluster' not in cluster_df.columns: missing_cols.append("cluster_df.cluster")
    if missing_cols:
        raise ValueError(f"Colonnes manquantes: {', '.join(missing_cols)}")

    return df, cluster_df                   

# Fonction pour convertir les colonnes de date
def convert_date_columns(df):
    date_columns = ['event_date_time', 'Date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df

# === après feature_engineering(...) et le merge features_df, pour chaque split ===
def memory_safe_numeric_impute(train_df, test_df):
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    # union: colonnes avec NaN dans train OU dans test
    nan_cols = [c for c in num_cols
                if train_df[c].isna().any() or test_df[c].isna().any()]

    if not nan_cols:
        return train_df, test_df

    # nettoyer inf/-inf -> NaN
    for df in (train_df, test_df):
        df[nan_cols] = df[nan_cols].replace([np.inf, -np.inf], np.nan)

    # downcast float
    for c in nan_cols:
        if pd.api.types.is_float_dtype(train_df[c]):
            train_df[c] = pd.to_numeric(train_df[c], downcast='float')
        if pd.api.types.is_float_dtype(test_df[c]):
            test_df[c]  = pd.to_numeric(test_df[c],  downcast='float')

    # médianes sur le TRAIN uniquement
    med = train_df[nan_cols].median()

    # imputation
    train_df[nan_cols] = train_df[nan_cols].fillna(med)
    test_df[nan_cols]  = test_df[nan_cols].fillna(med)

    return train_df, test_df


# Fonction pour analyser les données manquantes
def analyze_missing_data(df, cluster_id):
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing_Values': missing_data,
        'Percentage': missing_percent
    }).sort_values('Percentage', ascending=False)
    
    # Sauvegarder les informations sur les données manquantes
    missing_df.to_csv(f"TimeSeries_forecasting/cluster_{cluster_id}_missing_data.csv")
    
    # Visualiser les données manquantes
    plt.figure(figsize=(12, 8))
    plt.title(f'Pourcentage de données manquantes par colonne - Cluster {cluster_id}')
    sns.barplot(x=missing_df.index[missing_df['Percentage'] > 0], 
                y=missing_df['Percentage'][missing_df['Percentage'] > 0])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"TimeSeries_forecasting/cluster_{cluster_id}_missing_data.png")
    plt.close()
    
    return missing_df

# Fonction pour traiter les données manquantes
def handle_missing_data(df, cluster_id):
    print(f"Traitement des données manquantes pour le cluster {cluster_id}...")
    
    # Analyser les données manquantes
    missing_df = analyze_missing_data(df, cluster_id)
    
    # Colonnes numériques et catégorielles
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Stratégie pour les colonnes numériques avec peu de valeurs manquantes (<5%)
    for col in numeric_cols:
        missing_pct = missing_df.loc[col, 'Percentage'] if col in missing_df.index else 0
        
        if missing_pct > 0 and missing_pct < 5:
            # Utiliser la médiane pour les colonnes avec peu de valeurs manquantes
            df[col] = df[col].fillna(df[col].median())
        elif missing_pct >= 5 and missing_pct < 15:
            # Utiliser KNN pour les colonnes avec un nombre modéré de valeurs manquantes
            imputer = KNNImputer(n_neighbors=5)
            df[col] = imputer.fit_transform(df[[col]])[:, 0]
        elif missing_pct >= 15:
            # Pour les colonnes avec beaucoup de valeurs manquantes, créer un indicateur
            df[f"{col}_missing"] = df[col].isnull().astype(int)
            # Puis imputer avec la médiane
            df[col] = df[col].fillna(df[col].median())
    
    # Stratégie pour les colonnes catégorielles
    for col in categorical_cols:
        if col in missing_df.index and missing_df.loc[col, 'Missing_Values'] > 0:
            # Créer une catégorie "Unknown" pour les valeurs manquantes
            df[col] = df[col].fillna("Unknown")
    
    return df


def make_ohe():
    # scikit-learn 1.2+ : 'sparse_output'; avant : 'sparse'
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def prepare_features_df(features_df, k_max=256):
    f = features_df.copy()

    # gardons ci_name + numériques seulement
    num_cols = f.select_dtypes(include=[np.number]).columns.tolist()
    cols = ['ci_name'] + num_cols
    f = f.loc[:, [c for c in cols if c in f.columns]]

    # une seule ligne par ci_name (si plusieurs lignes existent)
    if f['ci_name'].duplicated().any():
        f = f.groupby('ci_name', as_index=False).mean(numeric_only=True)

    # supprimer colonnes constantes / toutes NaN
    nunique = f.nunique(dropna=False)
    drop_const = nunique[(nunique <= 1) & (nunique.index != 'ci_name')].index.tolist()
    f = f.drop(columns=drop_const)

    # prioriser par variance et limiter à k_max
    num_cols = f.select_dtypes(include=[np.number]).columns
    if len(num_cols) > k_max:
        var = f[num_cols].var().sort_values(ascending=False)
        topk = var.head(k_max).index.tolist()
        f = f[['ci_name'] + topk]

    # downcast en float32 pour diviser la mémoire par ~2
    for c in f.select_dtypes(include=[np.number]).columns:
        f[c] = pd.to_numeric(f[c], downcast='float')

    # préfixe pour éviter collisions de noms
    ren = {c: f'fx_{c}' for c in f.columns if c not in ['ci_name']}
    f = f.rename(columns=ren)

    return f


def feature_engineering(df, cluster_id, light=True):
    print(f"Feature engineering pour le cluster {cluster_id}...")

    # ⚠️ éviter la copie intégrale : df = df.copy() fait un double de tout
    # garde df tel quel, ou fais une copie seulement des colonnes que tu crées
    # df = df.copy()  # -> commenter pour limiter les pics mémoire

    # 0) dtypes compacts dès le début
    for c in df.select_dtypes(include=['float64']).columns:
        df[c] = pd.to_numeric(df[c], downcast='float')
    for c in df.select_dtypes(include=['int64']).columns:
        # attention: si des NaN possibles, garde float32; int32 ne supporte pas NaN
        if not df[c].isna().any():
            df[c] = pd.to_numeric(df[c], downcast='integer')

    # 1) Features temporelles
    if 'event_date_time' in df.columns:
        edt = df['event_date_time']
        df['Year']        = edt.dt.year.astype('int16')
        df['Month']       = edt.dt.month.astype('int8')
        df['Quarter']     = edt.dt.quarter.astype('int8')
        df['Day_of_year'] = edt.dt.dayofyear.astype('int16')
        df['Week_of_year']= edt.dt.isocalendar().week.astype('int16')
        df['Hour']        = edt.dt.hour.astype('int8')
        df['Is_weekend']  = (edt.dt.dayofweek >= 5).astype('int8')

        # cycliques en float32
        df['Month_sin'] = np.sin(2*np.pi*df['Month']/12.).astype('float32')
        df['Month_cos'] = np.cos(2*np.pi*df['Month']/12.).astype('float32')

        df['Is_month_start']  = (edt.dt.day <= 5).astype('int8')
        df['Is_month_middle'] = ((edt.dt.day > 5) & (edt.dt.day <= 25)).astype('int8')
        df['Is_month_end']    = (edt.dt.day > 25).astype('int8')
        df['Is_week_start']   = (edt.dt.dayofweek == 0).astype('int8')
        df['Is_week_end']     = (edt.dt.dayofweek == 4).astype('int8')

        # Part_of_day -> garde la variable catégorielle; PAS de get_dummies ici
        df['Part_of_day'] = pd.cut(
            df['Hour'], bins=[0,6,12,18,24],
            labels=['Night','Morning','Afternoon','Evening'],
            right=True, include_lowest=True
        )

    # 2) Lags & rollings — version “light”
    if 'event_value' in df.columns and 'ci_name' in df.columns:
        df.sort_values(['ci_name','event_date_time'], inplace=True)

        # Lags essentiels uniquement (réduis la liste si besoin)
        for lag in [1, 3, 12, 24]:
            df[f'Lag_{lag}'] = df.groupby('ci_name')['event_value'].shift(lag).astype('float32')

        # Diff simple
        df['Diff_1'] = (df['event_value'] - df['Lag_1']).astype('float32')

        # Rollings : limite-toi à 2 fenêtres et 2 stats (mean/std)
        windows = [6, 24] if light else [3, 6, 12, 24]
        for w in windows:
            # transform -> évite les MultiIndex temporaires de groupby.rolling
            mean_w = df.groupby('ci_name')['event_value'].transform(
                lambda s: s.rolling(window=w, min_periods=2).mean()
            ).astype('float32')
            std_w = df.groupby('ci_name')['event_value'].transform(
                lambda s: s.rolling(window=w, min_periods=2).std()
            ).astype('float32')

            df[f'Rolling_mean_{w}'] = mean_w
            df[f'Rolling_std_{w}']  = std_w

        # Déviations/z-score avec la fenêtre la plus “lente” existante
        base_w = 24 if 24 in windows else windows[-1]
        df['Deviation_from_mean'] = (df['event_value'] - df[f'Rolling_mean_{base_w}']).astype('float32')
        df['Z_score'] = (df['Deviation_from_mean'] / (df[f'Rolling_std_{base_w}'] + 1e-6)).astype('float32')

        # Trend simple
        if all(f'Rolling_mean_{w}' in df.columns for w in [6, 24]):
            df['Trend_up'] = (df['Rolling_mean_6'] > df['Rolling_mean_24']).astype('int8')
            df['Trend_down'] = (df['Rolling_mean_6'] < df['Rolling_mean_24']).astype('int8')

    # 3) Heures d'ouverture
    if 'site_business_hour' in df.columns and 'Hour' in df.columns:
        # vectorisé et int8
        sbh = df['site_business_hour'].astype(str).str.split('-', expand=True)
        try:
            start = pd.to_numeric(sbh[0], errors='coerce').fillna(-1)
            end   = pd.to_numeric(sbh[1], errors='coerce').fillna(-1)
            df['Is_business_hour'] = ((start <= df['Hour']) & (df['Hour'] <= end)).astype('int8')
        except Exception:
            df['Is_business_hour'] = 0

    # 4) Interactions légères
    if 'Is_weekend' in df.columns and 'Hour' in df.columns:
        df['Weekend_hour'] = (df['Is_weekend'] * df['Hour']).astype('int16')

    if 'site_criticallity' in df.columns and 'Hour' in df.columns:
        criticality_map = {'low':1,'medium':2,'high':3,'very high':4}
        df['criticality_numeric'] = df['site_criticallity'].map(criticality_map).fillna(0).astype('int8')
        df['Criticality_hour'] = (df['criticality_numeric'] * df['Hour']).astype('int16')

    # 5) Plot : désactive si trop gros (sinon seaborn alloue beaucoup)
    try:
        nrows = len(df)
        if nrows <= 300_000:  # seuil de sécurité
            numeric_features = df.select_dtypes(include=['float32','float64','int8','int16','int32','uint8']).columns
            plt.figure(figsize=(15,10))
            for i, feature in enumerate(list(numeric_features[:16])):
                plt.subplot(4,4,i+1)
                sns.histplot(df[feature].dropna().sample(min(100_000, df[feature].notna().sum()), random_state=42), kde=False)
                plt.title(feature)
            plt.tight_layout()
            plt.savefig(f"TimeSeries_forecasting/cluster_{cluster_id}_feature_distributions.png")
            plt.close()
        else:
            print(f"Plots ignorés (n={nrows:,})")
    except Exception as e:
        print(f"Impossible de générer les distributions des features: {e}")

    return df



def select_features(X_train, y_train, X_test, cluster_id,
                    max_rows=200_000,     # nombre max de lignes pour la sélection
                    max_cols_corr=150,    # nombre max de colonnes pour la corrélation pairwise/heatmap
                    rf_trees=50):         # n_estimators RF pour importance

    print(f"Sélection des features pour le cluster {cluster_id}...")

    # 0) Downcast pour réduire la mémoire
    def _downcast_df(df):
        for c in df.select_dtypes(include=['float64']).columns:
            df[c] = pd.to_numeric(df[c], downcast='float')
        for c in df.select_dtypes(include=['int64']).columns:
            if not df[c].isna().any():
                df[c] = pd.to_numeric(df[c], downcast='integer')
        return df

    X_train = _downcast_df(X_train)
    X_test  = _downcast_df(X_test)
    # y en float32
    y_train = pd.Series(pd.to_numeric(y_train, downcast='float'), index=y_train.index)

    # 1) Sous-échantillonnage lignes pour calculs lourds
    if len(X_train) > max_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), size=max_rows, replace=False)
        Xs = X_train.iloc[idx].copy()
        ys = y_train.iloc[idx].copy()
    else:
        Xs, ys = X_train, y_train

    # 2) Retirer colonnes constantes/NA-only AVANT tout
    nunique = Xs.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        Xs.drop(columns=const_cols, inplace=True)
        X_train = X_train.drop(columns=const_cols, errors='ignore')
        X_test  = X_test.drop(columns=const_cols,  errors='ignore')

    # 3) Corrélation simple avec la cible (sur Xs, ys)
    corr_scores = {}
    for col in Xs.columns:
        s = Xs[col]
        if np.issubdtype(s.dtype, np.number):
            try:
                v = np.corrcoef(s.fillna(s.median()), ys)[0, 1]
                if np.isfinite(v):
                    corr_scores[col] = abs(v)
            except Exception:
                pass
    correlation_df = (pd.DataFrame({'Feature': list(corr_scores.keys()),
                                    'Correlation': list(corr_scores.values())})
                      .sort_values('Correlation', ascending=False))
    correlation_df.to_csv(f"TimeSeries_forecasting/cluster_{cluster_id}_correlation_scores.csv", index=False)

    # 4) Sélection provisoire par corrélation (réduit l’espace)
    top_corr_keep = set(correlation_df.head(max_cols_corr * 2)['Feature'])  # un peu large
    Xs_small = Xs[list(top_corr_keep)]

    # 5) Matrice de corrélation pairwise (sur max_cols_corr pour éviter O(p²) massif)
    corr_cols = list(correlation_df.head(max_cols_corr)['Feature'])
    corr_matrix = Xs_small[corr_cols].corr()
    # heatmap facultatif (peut être lourd) : on garde mais sans annot et taille raisonnable
    plt.figure(figsize=(min(20, 0.2*len(corr_cols)+6), min(16, 0.2*len(corr_cols)+6)))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                linewidths=0.2, vmin=-1, vmax=1)
    plt.title(f'Matrice de Corrélation (subset) - Cluster {cluster_id}')
    plt.tight_layout()
    plt.savefig(f"TimeSeries_forecasting/cluster_{cluster_id}_correlation_matrix.png")
    plt.close()

    # 6) Paires très corrélées (|corr| > 0.95) sur le subset
    high_corr_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            cij = corr_matrix.iloc[i, j]
            if abs(cij) > 0.95:
                high_corr_pairs.append((cols[i], cols[j], cij))
    high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature1', 'Feature2', 'Correlation'])
    high_corr_df.to_csv(f"TimeSeries_forecasting/cluster_{cluster_id}_high_correlation_pairs.csv", index=False)

    # 7) RandomForest importance (sur Xs_small) — version LIGHT
    #    -> n_jobs=1 (évite la duplication mémoire par worker)
    #    -> bootstrap=True + max_samples (subsample par arbre)
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(
        n_estimators=rf_trees,
        max_depth=12,
        min_samples_leaf=5,
        max_features='sqrt',
        bootstrap=True,
        max_samples=0.25,   # 25% des lignes par arbre
        random_state=42,
        n_jobs=1
    )
    rf.fit(Xs_small.fillna(Xs_small.median()), ys)
    rf_imp = (pd.DataFrame({'Feature': Xs_small.columns, 'Importance': rf.feature_importances_})
                .sort_values('Importance', ascending=False))
    rf_imp.to_csv(f"TimeSeries_forecasting/cluster_{cluster_id}_feature_importance.csv", index=False)

    # 8) Information mutuelle (sur Xs_small)
    from sklearn.feature_selection import mutual_info_regression
    mi_scores = mutual_info_regression(Xs_small.fillna(Xs_small.median()), ys, random_state=42)
    mi_df = (pd.DataFrame({'Feature': Xs_small.columns, 'MI_Score': mi_scores})
                .sort_values('MI_Score', ascending=False))
    mi_df.to_csv(f"TimeSeries_forecasting/cluster_{cluster_id}_mutual_info_scores.csv", index=False)

    # 9) Combinaison des scores (sur l’intersection des colonnes évaluées)
    common = list(set(Xs_small.columns) & set(correlation_df['Feature']))
    combined = pd.DataFrame({'Feature': common})
    combined = combined.merge(correlation_df, on='Feature', how='left')
    combined = combined.merge(rf_imp, on='Feature', how='left')
    combined = combined.merge(mi_df, on='Feature', how='left')

    for col in ['Correlation', 'Importance', 'MI_Score']:
        if col in combined.columns and combined[col].notna().any():
            m = combined[col].max()
            if m and np.isfinite(m) and m > 0:
                combined[f'{col}_norm'] = combined[col] / m
            else:
                combined[f'{col}_norm'] = 0.0
        else:
            combined[f'{col}_norm'] = 0.0

    score_cols = [c for c in combined.columns if c.endswith('_norm')]
    combined['Combined_Score'] = combined[score_cols].mean(axis=1)
    combined = combined.sort_values('Combined_Score', ascending=False)
    combined.to_csv(f"TimeSeries_forecasting/cluster_{cluster_id}_combined_feature_scores.csv", index=False)

    # 10) Sélection finale : top-K raisonnable (ou seuil)
    K = 100  # <- ajuste si besoin
    top_features = combined.head(K)['Feature'].tolist()

    # 11) Retirer les paires hautement corrélées (sur le subset)
    features_to_drop = set()
    comb_map = dict(zip(combined['Feature'], combined['Combined_Score']))
    for _, row in high_corr_df.iterrows():
        f1, f2 = row['Feature1'], row['Feature2']
        if f1 in top_features and f2 in top_features:
            s1 = comb_map.get(f1, 0)
            s2 = comb_map.get(f2, 0)
            if s1 >= s2:
                features_to_drop.add(f2)
            else:
                features_to_drop.add(f1)
    final_features = [f for f in top_features if f not in features_to_drop]

    print(f"Nombre de features sélectionnées pour le cluster {cluster_id}: {len(final_features)}")
    pd.DataFrame({'Feature': final_features}).to_csv(
        f"TimeSeries_forecasting/cluster_{cluster_id}_final_features.csv", index=False)

    # 12) Filtrer les matrices complètes
    X_train_filtered = X_train[final_features]
    X_test_filtered  = X_test[final_features]
    return X_train_filtered, X_test_filtered, final_features


# Fonction pour l'optimisation des hyperparamètres avec Optuna
def optimize_model(X_train, y_train, model_name, cluster_id):
    print(f"Optimisation du modèle {model_name} pour le cluster {cluster_id}...")

    # 0) Sous-échantillonnage lignes pour l'optim (évite de tuner sur 3.3M lignes)
    n_rows = len(X_train)
    if n_rows > TUNE_MAX_ROWS:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_rows, size=TUNE_MAX_ROWS, replace=False)
        X_tune = X_train.iloc[idx].copy()
        y_tune = y_train.iloc[idx].copy()
    else:
        X_tune, y_tune = X_train, y_train

    # 1) Choix du nombre de folds
    n_splits = CV_FOLDS_BIG if n_rows >= BIG_CLUSTER_ROWS else CV_FOLDS_SMALL

    is_big = len(X_train) >= BIG_CLUSTER_ROWS


    # 2) Early stopping pour boosters (si possible)
    def fit_with_es(model, Xtr, ytr, Xva, yva):
        if isinstance(model, XGBRegressor):
            model.set_params(n_estimators=500, eval_metric='rmse', verbosity=0)
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], early_stopping_rounds=50, verbose=False)
        elif isinstance(model, LGBMRegressor):
            model.set_params(n_estimators=1000)
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[
                # early stopping à ~50 itérations sans gain
                # (compat scikit lightgbm>=3.3)
                # lgb.early_stopping(50) si tu utilises l'API native
            ])
        else:
            model.fit(Xtr, ytr)

    # 3) Court-circuit pour les modèles sans hyperparamètres
    if model_name == 'LinearRegression':
        model = LinearRegression()
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for tr, va in tscv.split(X_tune):
            Xtr, Xva = X_tune.iloc[tr], X_tune.iloc[va]
            ytr, yva = y_tune.iloc[tr], y_tune.iloc[va]
            fit_with_es(model, Xtr, ytr, Xva, yva)
            pred = model.predict(Xva)
            scores.append(np.sqrt(mean_squared_error(yva, pred)))
        best_rmse = float(np.mean(scores))
        model.fit(X_train, y_train)
        return model, {'model': model_name, 'best_params': {}, 'best_rmse': best_rmse}

    # 4) Définir l'espace de recherche
        # 4) Définir l'espace de recherche
    def objective(trial, Xs=X_tune, ys=y_tune):
        # IMPORTANT : capturer X_tune/y_tune pour éviter NameError
        # et s'assurer qu'on utilise bien le sous-échantillon de tuning.

        # modèles linéaires à scaler
        linear_like = ['Ridge', 'Lasso', 'ElasticNet']

        from sklearn.pipeline import Pipeline

        # === Définition du modèle (identique à celle que je t'ai donnée) ===
        if model_name == 'Ridge':
            if is_big:
                base = SGDRegressor(
                    loss='squared_error', penalty='l2',
                    alpha=trial.suggest_float('alpha', 1e-6, 1e-2, log=True),
                    learning_rate='optimal',
                    early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=5, max_iter=2000, tol=1e-4,
                    random_state=42
                )
            else:
                base = Ridge(alpha=trial.suggest_float('alpha', 1e-3, 10.0, log=True))
            model = Pipeline([('scaler', StandardScaler()), ('est', base)])

        elif model_name == 'Lasso':
            if is_big:
                base = SGDRegressor(
                    loss='squared_error', penalty='l1',
                    alpha=trial.suggest_float('alpha', 1e-6, 1e-2, log=True),
                    learning_rate='optimal',
                    early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=5, max_iter=2000, tol=1e-4,
                    random_state=42
                )
            else:
                base = Lasso(alpha=trial.suggest_float('alpha', 1e-4, 1.0, log=True), max_iter=50_000)
            model = Pipeline([('scaler', StandardScaler()), ('est', base)])

        elif model_name == 'ElasticNet':
            if is_big:
                base = SGDRegressor(
                    loss='squared_error', penalty='elasticnet',
                    alpha=trial.suggest_float('alpha', 1e-6, 1e-2, log=True),
                    l1_ratio=trial.suggest_float('l1_ratio', 0.0, 1.0),
                    learning_rate='optimal',
                    early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=5, max_iter=2000, tol=1e-4,
                    random_state=42
                )
            else:
                base = ElasticNet(
                    alpha=trial.suggest_float('alpha', 1e-4, 1.0, log=True),
                    l1_ratio=trial.suggest_float('l1_ratio', 0.0, 1.0),
                    max_iter=50_000
                )
            model = Pipeline([('scaler', StandardScaler()), ('est', base)])

        elif model_name == 'RandomForest':
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int('n_estimators', 100, 300),
                max_depth=trial.suggest_int('max_depth', 8, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                bootstrap=True, max_samples=0.2, n_jobs=1, random_state=42
            )

        elif model_name == 'GradientBoosting':
            model = GradientBoostingRegressor(
                n_estimators=trial.suggest_int('n_estimators', 100, 400),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                max_depth=trial.suggest_int('max_depth', 2, 6),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                random_state=42
            )

        elif model_name == 'XGBoost':
            model = XGBRegressor(
                n_estimators=trial.suggest_int('n_estimators', 200, 1200),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                reg_alpha=trial.suggest_float('reg_alpha', 0.0, 1.0),
                reg_lambda=trial.suggest_float('reg_lambda', 0.0, 1.0),
                random_state=42, tree_method='hist', n_jobs=1
            )

        elif model_name == 'LightGBM':
            # éviter max_depth=0 — on force -1 (illimité) ou >=3
            depth = trial.suggest_categorical('max_depth', [-1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            model = LGBMRegressor(
                n_estimators=trial.suggest_int('n_estimators', 300, 2000),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                max_depth=depth,
                num_leaves=trial.suggest_int('num_leaves', 31, 255),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                reg_alpha=trial.suggest_float('reg_alpha', 0.0, 1.0),
                reg_lambda=trial.suggest_float('reg_lambda', 0.0, 1.0),
                random_state=42, n_jobs=1
            )

        elif model_name == 'CatBoost':
            model = CatBoostRegressor(
                iterations=trial.suggest_int('iterations', 300, 2000),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                depth=trial.suggest_int('depth', 4, 10),
                l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
                random_seed=42, verbose=False
            )

        elif model_name == 'SVR':
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('est', SVR(
                    C=trial.suggest_float('C', 0.1, 50.0, log=True),
                    epsilon=trial.suggest_float('epsilon', 0.01, 1.0, log=True),
                    gamma=trial.suggest_categorical('gamma', ['scale', 'auto'])
                ))
            ])
        else:
            raise ValueError(f"Modèle {model_name} non supporté")

        # === CV temporelle ===
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for fold, (tr, va) in enumerate(tscv.split(Xs)):
            if isinstance(Xs, np.ndarray):
                Xtr, Xva = Xs[tr], Xs[va]
            else:
                Xtr, Xva = Xs.iloc[tr], Xs.iloc[va]
            ytr, yva = ys.iloc[tr], ys.iloc[va]

            # clone pour éviter fuites d’état entre folds (early stopping, etc.)
            model_fold = clone(model)
            model_fold.fit(Xtr, ytr)
            pred = model_fold.predict(Xva)

            rmse = float(np.sqrt(mean_squared_error(yva, pred)))
            scores.append(rmse)
            trial.report(rmse, step=fold)

        return float(np.mean(scores))



    # 5) Étude Optuna avec pruner + trials adaptés + TIMEOUT (ex. 30 min par modèle)
    TIMEOUT_PER_MODEL = 30 * 60  # 30 minutes
    study = optuna.create_study(direction='minimize', pruner=PRUNER)
    study.optimize(objective,
                n_trials=TRIALS.get(model_name, 50),
                timeout=TIMEOUT_PER_MODEL)


    best_params = study.best_params
    best_value  = study.best_value

        # 6) Re-fit final sur TOUT le train avec les meilleurs hyperparams
    from sklearn.pipeline import Pipeline

    def make_best_model(name, params):
        if name == 'Ridge':
            base = Ridge(**params)
            return Pipeline([('scaler', StandardScaler()), ('est', base)])
        if name == 'Lasso':
            base = Lasso(max_iter=50_000, **params)
            return Pipeline([('scaler', StandardScaler()), ('est', base)])
        if name == 'ElasticNet':
            base = ElasticNet(max_iter=50_000, **params)
            return Pipeline([('scaler', StandardScaler()), ('est', base)])
        if name == 'RandomForest':
            return RandomForestRegressor(n_jobs=1, random_state=42, **params)
        if name == 'GradientBoosting':
            return GradientBoostingRegressor(random_state=42, **params)
        if name == 'XGBoost':
            return XGBRegressor(random_state=42, tree_method='hist', n_jobs=1, **params)
        if name == 'LightGBM':
            return LGBMRegressor(random_state=42, n_jobs=1, **params)
        if name == 'CatBoost':
            return CatBoostRegressor(random_seed=42, verbose=False, **params)
        if name == 'SVR':
            return Pipeline([('scaler', StandardScaler()), ('est', SVR(**params))])
        raise ValueError(name)

    if model_name in ['Ridge','Lasso','ElasticNet'] and is_big:
        # version SGD + pipeline pour les très gros volumes
        if model_name == 'Ridge':
            base = SGDRegressor(loss='squared_error', penalty='l2',
                                early_stopping=False, max_iter=2000, tol=1e-4,
                                random_state=42, **best_params)
        elif model_name == 'Lasso':
            base = SGDRegressor(loss='squared_error', penalty='l1',
                                early_stopping=False, max_iter=2000, tol=1e-4,
                                random_state=42, **best_params)
        else:  # ElasticNet
            base = SGDRegressor(loss='squared_error', penalty='elasticnet',
                                early_stopping=False, max_iter=2000, tol=1e-4,
                                random_state=42, **best_params)
        best_model = Pipeline([('scaler', StandardScaler()), ('est', base)])
    else:
        best_model = make_best_model(model_name, best_params)

    best_model.fit(X_train, y_train)
    results = {'model': model_name, 'best_params': best_params, 'best_rmse': best_value}
    return best_model, results


# Fonction pour évaluer les modèles
def evaluate_model(model, X_test, y_test, model_name, cluster_id):
    print(f"Évaluation du modèle {model_name} pour le cluster {cluster_id}...")
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Métriques d'évaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = safe_mape(y_test, y_pred)
    smape_val = smape(y_test, y_pred)

    
    # Sauvegarder les métriques
    metrics = {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'smape' : smape_val
    }
    
    # Visualiser les prédictions vs réalité
    plt.figure(figsize=(12, 6))
    
    # Échantillonner pour la lisibilité si nécessaire
    sample_size = min(1000, len(y_test))
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    plt.scatter(y_test.iloc[indices], y_pred[indices], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    plt.title(f'Prédictions vs Réalité - {model_name} - Cluster {cluster_id}')
    plt.savefig(f"TimeSeries_forecasting/cluster_{cluster_id}_{model_name}_predictions_vs_actual.png")
    plt.close()
    
    # Visualiser les résidus
    residuals = y_test - y_pred
    
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Prédictions')
    plt.ylabel('Résidus')
    plt.title(f'Résidus - {model_name} - Cluster {cluster_id}')
    plt.savefig(f"TimeSeries_forecasting/cluster_{cluster_id}_{model_name}_residuals.png")
    plt.close()
    
    # Distribution des résidus
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Résidus')
    plt.ylabel('Fréquence')
    plt.title(f'Distribution des résidus - {model_name} - Cluster {cluster_id}')
    plt.savefig(f"TimeSeries_forecasting/cluster_{cluster_id}_{model_name}_residuals_distribution.png")
    plt.close()
    
    # Visualiser les prédictions dans le temps pour quelques équipements
    if 'ci_name' in X_test.index.names:
        # Si les données sont indexées par ci_name
        unique_cis = X_test.index.get_level_values('ci_name').unique()
        sample_cis = np.random.choice(unique_cis, min(5, len(unique_cis)), replace=False)
        
        plt.figure(figsize=(15, 10))
        for i, ci in enumerate(sample_cis):
            plt.subplot(len(sample_cis), 1, i+1)
            
            ci_data = X_test[X_test.index.get_level_values('ci_name') == ci]
            ci_y_test = y_test[y_test.index.get_level_values('ci_name') == ci]
            ci_y_pred = model.predict(ci_data)
            
            plt.plot(ci_y_test.index.get_level_values('event_date_time'), ci_y_test.values, label='Réel')
            plt.plot(ci_y_test.index.get_level_values('event_date_time'), ci_y_pred, label='Prédit')
            plt.title(f'Équipement: {ci}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"TimeSeries_forecasting/cluster_{cluster_id}_{model_name}_time_series_predictions.png")
        plt.close()
    
    return metrics

# Fonction principale pour traiter chaque cluster
def process_cluster(df, cluster_id, features_df=None):
    print(f"\n{'='*50}")
    print(f"Traitement du cluster {cluster_id}")
    print(f"{'='*50}")
    
    # Filtrer les données pour ce cluster
    cluster_data = df[df['cluster'] == cluster_id].copy()
    
    # 1. Séparation train/test (les 3 dernières semaines pour le test)
    max_date = cluster_data['event_date_time'].max()
    test_start_date = max_date - timedelta(days=21)  # 3 semaines
    
    train_data = cluster_data[cluster_data['event_date_time'] < test_start_date]
    test_data = cluster_data[cluster_data['event_date_time'] >= test_start_date]
    
    print(f"Taille des données d'entraînement: {train_data.shape}")
    print(f"Taille des données de test: {test_data.shape}")
    
    # 2. Vérification et traitement des données manquantes
    train_data = handle_missing_data(train_data, cluster_id)
    test_data = handle_missing_data(test_data, cluster_id)
    
    # 3. Feature engineering
    train_data = feature_engineering(train_data, cluster_id)
    # après le feature engineering, côté TRAIN uniquement
    max_lag = 24
    train_data = (train_data
                .sort_values(['ci_name','event_date_time'])
                .groupby('ci_name', group_keys=False)
                .apply(lambda g: g.iloc[max_lag:]))  # drop les 1ères lignes insuffisantes
    test_data = feature_engineering(test_data, cluster_id)
    
    # Si features_df est disponible, fusionner avec les données
    if features_df is not None:
        features_df = prepare_features_df(features_df, k_max=256)  # ajuste k_max si besoin
        print("features_df compact:", features_df.shape)  # debug
        train_data = pd.merge(train_data, features_df, on='ci_name', how='left')
        test_data  = pd.merge(test_data,  features_df, on='ci_name', how='left')
    
    # juste après feature_engineering(...) et la fusion éventuelle
    num_cols = train_data.select_dtypes(include=[np.number]).columns
    train_data, test_data = memory_safe_numeric_impute(train_data, test_data)


    
        # === Préparation des features et de la cible
    target_col = 'event_value'
    exclude_cols = [
        'event_value', 'event_date_time', 'Date', 'cluster', 'ci_name',
        'site_name', 'site_city', 'site_country', 'site_business_hour',
        'service_line'
    ]
    feature_cols_raw = [c for c in train_data.columns if c not in exclude_cols]

    # Split cat/num d’après dtypes du train
    cat_cols = [c for c in feature_cols_raw if str(train_data[c].dtype) in ('object','category')]
    num_cols = [c for c in feature_cols_raw if c not in cat_cols]

    n_rows_train = len(train_data)

    if n_rows_train >= BIG_CLUSTER_ROWS:
        # MODE MÉMOIRE-FRIENDLY : pas d’OHE, on code les catégories
        for c in cat_cols:
            train_data[c] = train_data[c].astype('category').cat.codes.astype('int16')
            test_data[c]  = test_data[c].astype('category').cat.codes.astype('int16')

        feature_cols = num_cols + cat_cols  # tout numérique maintenant
        X_train = train_data[feature_cols].astype('float32')
        X_test  = test_data[feature_cols].astype('float32')
        y_train = train_data[target_col].copy()
        y_test  = test_data[target_col].copy()

    else:
        # ColumnTransformer : OHE + passthrough
        preproc = ColumnTransformer(
            transformers=[
                ('cat', make_ohe(), cat_cols),
                ('num', 'passthrough', num_cols),
            ],
            remainder='drop'
        )
        X_train_arr = preproc.fit_transform(train_data[cat_cols + num_cols])
        X_test_arr  = preproc.transform(test_data[cat_cols + num_cols])

        cat_feature_names = []
        if len(cat_cols) > 0:
            cat_feature_names = preproc.named_transformers_['cat'].get_feature_names_out(cat_cols)
        feature_names = np.concatenate([cat_feature_names, np.array(num_cols)], axis=0)

        X_train = pd.DataFrame(X_train_arr, columns=feature_names, index=train_data.index)
        X_test  = pd.DataFrame(X_test_arr,  columns=feature_names, index=test_data.index)
        y_train = train_data[target_col].copy()
        y_test  = test_data[target_col].copy()


    
    # 4. Sélection des features
    X_train, X_test, selected_features = select_features(X_train, y_train, X_test, cluster_id)

    # juste après:
    # X_train, X_test, selected_features = select_features(...)

    def sanitize_X(X_train, X_test):
        # remplace inf/-inf par NaN
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test  = X_test.replace([np.inf, -np.inf], np.nan)

        # médianes calculées sur TRAIN (numériques uniquement)
        med = X_train.median(numeric_only=True)

        # impute
        X_train = X_train.fillna(med)
        X_test  = X_test.fillna(med)

        # si des colonnes restent full-NaN (cas extrême), on les drop
        bad_cols = [c for c in X_train.columns if X_train[c].isna().all()] + \
                [c for c in X_test.columns  if X_test[c].isna().all()]
        bad_cols = sorted(set(bad_cols))
        if bad_cols:
            X_train = X_train.drop(columns=bad_cols, errors='ignore')
            X_test  = X_test.drop(columns=bad_cols, errors='ignore')

        return X_train, X_test

    X_train, X_test = sanitize_X(X_train, X_test)

    
    # 5. Optimisation et évaluation des modèles
    # choix des modèles selon la taille
    n_rows_train = len(X_train)
    if n_rows_train >= BIG_CLUSTER_ROWS:
        # MODE RAPIDE: pas de RF/GB/SVR sur millions de lignes
        models_to_try = [
            'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
            'LightGBM', 'XGBoost'   # boosters -> plus efficients
        ]
        print(f"[Info] Gros cluster (n={n_rows_train:,}) -> MODE RAPIDE: {models_to_try}")
    else:
        models_to_try = [
            'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
            'RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM'
        ]

    
    all_results = []
    best_model_info = {'rmse': float('inf')}
    
    for model_name in models_to_try:
        try:
            # Optimisation des hyperparamètres
            model, optim_results = optimize_model(X_train, y_train, model_name, cluster_id)
            
            # Évaluation du modèle
            eval_metrics = evaluate_model(model, X_test, y_test, model_name, cluster_id)
            
            # Combiner les résultats
            combined_results = {**optim_results, **eval_metrics}
            all_results.append(combined_results)
            
            # Vérifier si c'est le meilleur modèle jusqu'à présent
            if eval_metrics['rmse'] < best_model_info['rmse']:
                best_model_info = {
                    'model_name': model_name,
                    'model': model,
                    'rmse': eval_metrics['rmse'],
                    'mae': eval_metrics['mae'],
                    'r2': eval_metrics['r2'],
                    'mape': eval_metrics['mape']
                }
                
                # Sauvegarder le meilleur modèle
                joblib.dump(model, f"TimeSeries_forecasting/cluster_{cluster_id}_best_model.pkl")
                
        except Exception as e:
            print(f"Erreur lors de l'optimisation/évaluation du modèle {model_name}: {str(e)}")
    
    # Sauvegarder tous les résultats
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"TimeSeries_forecasting/cluster_{cluster_id}_all_models_results.csv", index=False)
    
    # Visualiser la comparaison des modèles
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='rmse', data=results_df)
    plt.title(f'Comparaison des modèles par RMSE - Cluster {cluster_id}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"TimeSeries_forecasting/cluster_{cluster_id}_model_comparison_rmse.png")
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='r2', data=results_df)
    plt.title(f'Comparaison des modèles par R² - Cluster {cluster_id}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"TimeSeries_forecasting/cluster_{cluster_id}_model_comparison_r2.png")
    plt.close()
    
    # Résumé des résultats
    print(f"\nRésumé des résultats pour le cluster {cluster_id}:")
    print(f"Meilleur modèle: {best_model_info['model_name']}")
    print(f"RMSE: {best_model_info['rmse']:.4f}")
    print(f"MAE: {best_model_info['mae']:.4f}")
    print(f"R²: {best_model_info['r2']:.4f}")
    print(f"MAPE: {best_model_info['mape']:.2f}%")


    # === Sauvegarde des prédictions du cluster pour le graphe final ===
    # (on sauvegarde les prédictions sur le test set, alignées avec les colonnes d'intérêt)
    try:
        y_pred_full = best_model_info['model'].predict(X_test)
        preds_df = test_data[['ci_name', 'event_date_time', 'event_value']].copy()
        preds_df['y_pred'] = y_pred_full
        preds_df['cluster'] = cluster_id
        # sécurité: tri par temps
        preds_df = preds_df.sort_values(['ci_name', 'event_date_time'])
        preds_df.to_csv(f"TimeSeries_forecasting/cluster_{cluster_id}_predictions.csv", index=False)
    except Exception as e:
        print(f"[WARN] Impossible de sauvegarder les prédictions du cluster {cluster_id}: {e}")


    
    return best_model_info


def plot_final_10_devices(time_col='event_date_time',
                          real_col='event_value',
                          pred_col='y_pred',
                          out_path="TimeSeries_forecasting/final_10_devices.png",
                          seed=42):
    """
    Charge tous les CSV de prédictions de clusters, prend 10 équipements distincts
    (répartis sur plusieurs clusters si possible), et trace Réel vs Prédit dans le temps.
    """
    import glob
    import matplotlib.dates as mdates

    files = sorted(glob.glob("TimeSeries_forecasting/cluster_*_predictions.csv"))
    if not files:
        print("[WARN] Aucun fichier de prédictions cluster_*_predictions.csv trouvé.")
        return

    # Concaténer toutes les prédictions
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f, parse_dates=[time_col])
            dfs.append(d)
        except Exception as e:
            print(f"[WARN] Skip {f}: {e}")
    if not dfs:
        print("[WARN] Aucun CSV valide pour les prédictions.")
        return

    all_preds = pd.concat(dfs, ignore_index=True)

    # Liste (cluster, ci_name) uniques
    pairs = (all_preds[['cluster', 'ci_name']]
             .drop_duplicates()
             .sort_values(['cluster', 'ci_name']))
    clusters = pairs['cluster'].unique()
    n_clusters = len(clusters)

    # Échantillonnage stratifié: ~10/n_clusters par cluster (au moins 1)
    rng = np.random.default_rng(seed)
    per_cluster = max(1, int(np.ceil(10 / n_clusters)))
    sampled = []

    for cl in clusters:
        pool = pairs[pairs['cluster'] == cl]['ci_name'].values
        take = min(per_cluster, len(pool))
        if take > 0:
            chosen = rng.choice(pool, size=take, replace=False)
            sampled.extend([(cl, c) for c in chosen])

    # Si on a plus que 10, tronquer; si moins que 10, compléter au hasard global
    if len(sampled) > 10:
        sampled = sampled[:10]
    elif len(sampled) < 10:
        rest = [(int(r['cluster']), r['ci_name']) for _, r in pairs.iterrows()
                if (int(r['cluster']), r['ci_name']) not in sampled]
        need = min(10 - len(sampled), len(rest))
        if need > 0:
            sampled.extend(rng.choice(rest, size=need, replace=False).tolist())

    # Plot : 10 sous-graphiques
    n = len(sampled)
    if n == 0:
        print("[WARN] Impossible d’échantillonner des équipements.")
        return

    rows = 5 if n > 5 else n
    cols = 2 if n > 5 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.6*rows), sharex=False)
    if n == 1:
        axes = np.array([axes])  # uniformiser
    axes = axes.flatten()

    for i, (cl, ci) in enumerate(sampled):
        ax = axes[i]
        sub = (all_preds[(all_preds['cluster'] == cl) & (all_preds['ci_name'] == ci)]
            .sort_values(time_col))

        if sub.empty:
            ax.set_title(f"Cluster {cl} (aucune donnée)")
            continue

        # downsample pour lisibilité si très long (ex: garder 400 points max)
        if len(sub) > 400:
            step = int(np.ceil(len(sub) / 400))
            sub = sub.iloc[::step, :]

        ax.plot(sub[time_col], sub[real_col], label='Réel', linewidth=1.2)
        ax.plot(sub[time_col], sub[pred_col], label='Prédit', linewidth=1.2, alpha=0.9)

        # Titre simplifié : uniquement le cluster
        ax.set_title(f"Cluster {cl}")
        ax.set_ylabel("Bande passante")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        ax.grid(True, linestyle='--', alpha=0.3)

    # légende unique
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.suptitle("Évolution de la bande passante — Réel vs Prédit (10 équipements)", y=0.995)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Graphe final sauvegardé : {out_path}")



# Fonction principale
def main():
    # Charger les données
    df, cluster_df = load_data()
    features_df = None
    
    # Fusionner les données avec les clusters
    df = pd.merge(df, cluster_df, on='ci_name', how='inner')
    
    # Convertir les colonnes de date
    df = convert_date_columns(df)
    
    # Créer un résumé des clusters
    cluster_summary = df.groupby('cluster')['ci_name'].nunique().reset_index()
    cluster_summary.columns = ['Cluster', 'Nombre d\'équipements']
    cluster_summary.to_csv("TimeSeries_forecasting/cluster_summary.csv", index=False)
    
    print("Résumé des clusters:")
    print(cluster_summary)
    
    # Traiter chaque cluster
    all_cluster_results = []
    
    for cluster_id in df['cluster'].unique():
        cluster_result = process_cluster(df, cluster_id, features_df)
        all_cluster_results.append({
            'cluster': cluster_id,
            'best_model': cluster_result['model_name'],
            'rmse': cluster_result['rmse'],
            'mae': cluster_result['mae'],
            'r2': cluster_result['r2'],
            'mape': cluster_result['mape']
        })
    
    # Sauvegarder le résumé des résultats pour tous les clusters
    all_results_df = pd.DataFrame(all_cluster_results)
    all_results_df.to_csv("TimeSeries_forecasting/all_clusters_results.csv", index=False)
    
    # Visualiser la comparaison des performances entre clusters
    plt.figure(figsize=(10, 6))
    sns.barplot(x='cluster', y='rmse', data=all_results_df)
    plt.title('RMSE par cluster')
    plt.tight_layout()
    plt.savefig("TimeSeries_forecasting/clusters_comparison_rmse.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='cluster', y='r2', data=all_results_df)
    plt.title('R² par cluster')
    plt.tight_layout()
    plt.savefig("TimeSeries_forecasting/clusters_comparison_r2.png")
    plt.close()

    # ... après avoir traité tous les clusters et sauvegardé les CSV de prédictions
    print('Génération des 10 plots')
    plot_final_10_devices()

    
    print("\nTraitement terminé. Tous les résultats ont été sauvegardés dans le dossier 'TimeSeries_forecasting'.")

# Exécution du programme principal
if __name__ == "__main__":
    main()
