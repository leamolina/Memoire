import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, Birch, MeanShift, AffinityPropagation, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from scipy.stats import gmean
from sklearn_extra.cluster import KMedoids
import os
import warnings
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from kmodes.kmodes import KModes
import time
from tqdm import tqdm
import pycatch22
import numpy as np
import nolds
import pywt
from scipy import signal
import antropy as ant
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import scipy.signal as signal
from sklearn.impute import SimpleImputer
import pywt
import nolds
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Ignorer les avertissements
warnings.filterwarnings('ignore')

# Créer le dossier de résultats s'il n'existe pas
os.makedirs('results_catch22/Interface', exist_ok=True)
os.makedirs('results_catch22/Interface/feature_analysis', exist_ok=True)
os.makedirs('results_catch22/Interface/dimensionality_reduction', exist_ok=True)

# Charger les données
print("Chargement des données...")
df = pd.read_csv('Data/AllMonths/data_complete_interface.csv')
if not pd.api.types.is_datetime64_any_dtype(df['event_date_time']):
    print("Conversion de event_date_time en format datetime...")
    df['event_date_time'] = pd.to_datetime(df['event_date_time'])

# Afficher les informations sur le dataframe
print("Aperçu des données:")
print(df.head())
print("\nInformations sur les colonnes:")
print(df.info())
print("\nStatistiques descriptives:")
print(df.describe())

# Rajout des colonnes criticality
df['Criticality_unknown'] = (df['site_criticality'] == 'Unknown').astype(int)
print("value count ", df['Criticality_unknown'].value_counts())

# Vérifier les valeurs manquantes
print("\nValeurs manquantes par colonne:")
print(df.isnull().sum())

print("affiche le nombre de dates par ci_name :")
# Nombre de dates par ci_name
print(df.groupby('ci_name')['event_date_time'].nunique().sort_values(ascending=False))

def investigate_weekly_patterns(df):
    # Ajouter une colonne de semaine au DataFrame
    df['week'] = df['event_date_time'].dt.isocalendar().week

    # Compter combien de ci_name ont des données pour chaque semaine
    week_counts = df.groupby(['week'])['ci_name'].nunique()
    total_devices = df['ci_name'].nunique()

    print(f"Nombre total d'appareils: {total_devices}")
    print("Nombre et pourcentage d'appareils avec des données par semaine:")
    for week, count in week_counts.items():
        print(f"Semaine {week}: {count} appareils ({count/total_devices*100:.2f}%)")

    # Visualiser la distribution
    plt.figure(figsize=(15, 6))
    week_counts.plot(kind='bar')
    plt.axhline(y=total_devices, color='r', linestyle='-', label='Nombre total d\'appareils')
    plt.title('Nombre d\'appareils avec des données par semaine')
    plt.xlabel('Semaine')
    plt.ylabel('Nombre d\'appareils')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results_catch22/Interface/equipment_distribution_by_week.png', dpi=300)
    plt.close()

def visualize_device_consistency(df, devices_by_week, low_percentage_weeks):
    """
    Visualise la cohérence des appareils à travers les semaines.
    """
    if not low_percentage_weeks:
        print("Pas de semaines à faible pourcentage identifiées.")
        return
    
    # Créer une matrice de similarité
    similarity_matrix = np.zeros((len(low_percentage_weeks), len(low_percentage_weeks)))
    
    for i, week1 in enumerate(low_percentage_weeks):
        for j, week2 in enumerate(low_percentage_weeks):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                overlap = len(devices_by_week[week1].intersection(devices_by_week[week2]))
                similarity_matrix[i, j] = overlap / len(devices_by_week[week1])
    
    # Visualiser la matrice de similarité
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=low_percentage_weeks, yticklabels=low_percentage_weeks)
    plt.title("Similarité des appareils entre les semaines à faible pourcentage")
    plt.xlabel("Semaine")
    plt.ylabel("Semaine")
    plt.tight_layout()
    plt.savefig('results_catch22/Interface/device_similarity_matrix.png', dpi=300)
    plt.close()
    
    # Visualiser la distribution des appareils par semaine
    device_counts = {}
    all_devices = set()
    
    for week in low_percentage_weeks:
        all_devices.update(devices_by_week[week])
    
    for device in all_devices:
        device_counts[device] = [1 if device in devices_by_week[week] else 0 for week in low_percentage_weeks]
    
    # Convertir en DataFrame pour faciliter la visualisation
    device_df = pd.DataFrame.from_dict(device_counts, orient='index', columns=low_percentage_weeks)
    
    # Calculer le nombre de semaines où chaque appareil apparaît
    device_df['total_weeks'] = device_df.sum(axis=1)
    
    # Visualiser la distribution
    plt.figure(figsize=(10, 6))
    device_df['total_weeks'].value_counts().sort_index().plot(kind='bar')
    plt.title("Distribution des appareils par nombre de semaines")
    plt.xlabel("Nombre de semaines")
    plt.ylabel("Nombre d'appareils")
    plt.tight_layout()
    plt.savefig('results_catch22/Interface/device_week_distribution.png', dpi=300)
    plt.close()

def fill_nan(df):
    """
    Remplit les valeurs NaN dans site_country et site_business_hours en utilisant les valeurs
    du même site_name lorsque c'est possible, sinon utilise le mode.
    
    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données
        
    Returns:
        pandas.DataFrame: Le DataFrame avec les valeurs NaN remplies
    """
    print("Remplissage des valeurs manquantes dans site_country et site_business_hours...")
    
    # Créer une copie du DataFrame pour éviter les avertissements SettingWithCopyWarning
    df_filled = df.copy()
    
    # Vérifier les valeurs manquantes avant le remplissage
    print("Valeurs manquantes avant remplissage:")
    print(df_filled[['site_country', 'site_business_hours']].isnull().sum())
    if df_filled[['site_country', 'site_business_hours']].isnull().sum().sum() == 0:
        print("Aucune valeur manquante à remplir.")
        return df_filled    
    
    # 1. Remplir site_country en utilisant les valeurs du même site_name
    # Créer un dictionnaire de correspondance site_name -> site_country
    site_country_map = {}
    for name, group in df_filled.groupby('site_name'):
        # Prendre la valeur non-NaN la plus fréquente pour chaque site_name
        valid_countries = group['site_country'].dropna()
        if not valid_countries.empty:
            site_country_map[name] = valid_countries.mode()[0]
    
    # Appliquer le dictionnaire de correspondance
    for site_name, country in site_country_map.items():
        mask = (df_filled['site_name'] == site_name) & (df_filled['site_country'].isnull())
        df_filled.loc[mask, 'site_country'] = country
    
    # 2. Remplir site_business_hours en utilisant les valeurs du même site_name
    # Créer un dictionnaire de correspondance site_name -> site_business_hours
    site_hours_map = {}
    for name, group in df_filled.groupby('site_name'):
        # Prendre la valeur non-NaN la plus fréquente pour chaque site_name
        valid_hours = group['site_business_hours'].dropna()
        if not valid_hours.empty:
            site_hours_map[name] = valid_hours.mode()[0]
    
    # Appliquer le dictionnaire de correspondance
    for site_name, hours in site_hours_map.items():
        mask = (df_filled['site_name'] == site_name) & (df_filled['site_business_hours'].isnull())
        df_filled.loc[mask, 'site_business_hours'] = hours
    
    # 3. Remplir les valeurs manquantes restantes avec le mode
    # Pour site_country
    if df_filled['site_country'].isnull().any():
        mode_country = df_filled['site_country'].mode()[0]
        df_filled['site_country'] = df_filled['site_country'].fillna(mode_country)
        print(f"Valeurs manquantes restantes dans site_country remplies avec le mode: {mode_country}")
    
    # Pour site_business_hours
    if df_filled['site_business_hours'].isnull().any():
        mode_hours = df_filled['site_business_hours'].mode()[0]
        df_filled['site_business_hours'] = df_filled['site_business_hours'].fillna(mode_hours)
        print(f"Valeurs manquantes restantes dans site_business_hours remplies avec le mode: {mode_hours}")
    
    # Vérifier les valeurs manquantes après le remplissage
    print("Valeurs manquantes après remplissage:")
    print(df_filled[['site_country', 'site_business_hours']].isnull().sum())
    df_filled.to_csv("Data/AllMonths/data_complete_interface.csv")
    return df_filled

df = fill_nan(df)

print("Investigation des motifs hebdomadaires...")
investigate_weekly_patterns(df)

# Supprimer la colonne ci_type si elle est constante
if 'ci_type' in df.columns and df['ci_type'].nunique() == 1:
    df = df.drop('ci_type', axis=1)
    print("Colonne 'ci_type' supprimée car constante.")

# Convertir la colonne de date en datetime
df['event_date_time'] = pd.to_datetime(df['event_date_time'])

# Extraire des composantes temporelles supplémentaires
df['day_of_week'] = df['event_date_time'].dt.dayofweek
df['month'] = df['event_date_time'].dt.month
df['week_of_year'] = df['event_date_time'].dt.isocalendar().week

# Fonction pour extraire les caractéristiques catch22 et d'autres statistiques
def extract_time_series_features(group):
    group = group.copy()
    # Trier par date
    group = group.sort_values('event_date_time')
    
    # Normaliser event_value si nécessaire
    min_val = group['event_value'].min()
    max_val = group['event_value'].max()
    if max_val > min_val:
        group['event_value_norm'] = 100 * (group['event_value'] - min_val) / (max_val - min_val)
    else:
        group['event_value_norm'] = group['event_value']
    ts = group['event_value_norm'].values
    df_ts = pd.Series(ts)
    
    new_features = {}

    # 1) Dynamique & tendance
    new_features['hurst_exp'] = nolds.hurst_rs(ts)
    for w in [3, 6, 12, 24]:
        slopes = df_ts.rolling(w, min_periods=2).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
        )
        new_features[f'roll_slope_{w}h'] = slopes.mean()
        new_features[f'roll_std_{w}h'] = df_ts.rolling(w).std().mean()
        new_features[f'roll_mean_{w}h'] = df_ts.rolling(w).mean().mean()
        # Ajout: médiane et quantiles des fenêtres glissantes
        new_features[f'roll_median_{w}h'] = df_ts.rolling(w).median().mean()
        new_features[f'roll_q25_{w}h'] = df_ts.rolling(w).quantile(0.25).mean()
        new_features[f'roll_q75_{w}h'] = df_ts.rolling(w).quantile(0.75).mean()
        # Ajout: skewness et kurtosis des fenêtres glissantes
        new_features[f'roll_skew_{w}h'] = df_ts.rolling(w, min_periods=min(3, w)).skew().mean()
        new_features[f'roll_kurt_{w}h'] = df_ts.rolling(w, min_periods=min(4, w)).kurt().mean()
    
    diffs = np.diff(ts)
    new_features['mean_abs_diff'] = np.mean(np.abs(diffs))
    new_features['std_diff'] = np.std(diffs)
    # Ajout: statistiques sur les différences
    new_features['median_abs_diff'] = np.median(np.abs(diffs))
    new_features['max_abs_diff'] = np.max(np.abs(diffs))
    new_features['min_abs_diff'] = np.min(np.abs(diffs))
    new_features['diff_iqr'] = np.percentile(diffs, 75) - np.percentile(diffs, 25)
    
    # Ajout: différences d'ordre supérieur
    if len(ts) > 2:
        diff2 = np.diff(diffs)
        new_features['mean_abs_diff2'] = np.mean(np.abs(diff2))
        new_features['std_diff2'] = np.std(diff2)
        new_features['max_abs_diff2'] = np.max(np.abs(diff2))

    # 2) Fréquentiel
    freqs, psd = signal.welch(ts, fs=1.0)
    psd_norm = psd/psd.sum()
    new_features['spectral_entropy'] = -np.nansum(psd_norm*np.log2(psd_norm+1e-12))
    new_features['dominant_freq'] = freqs[np.argmax(psd)]
    new_features['spec_centroid'] = (freqs*psd_norm).sum()
    new_features['spec_bandwidth'] = np.sqrt(((freqs-new_features['spec_centroid'])**2*psd_norm).sum())
    
    # Ajout: caractéristiques spectrales supplémentaires
    if len(psd) > 1:
        # Énergie dans différentes bandes de fréquence
        low_freq_idx = freqs <= 0.1
        mid_freq_idx = (freqs > 0.1) & (freqs <= 0.3)
        high_freq_idx = freqs > 0.3
        
        new_features['low_freq_energy'] = np.sum(psd[low_freq_idx]) / np.sum(psd) if np.sum(psd) > 0 else 0
        new_features['mid_freq_energy'] = np.sum(psd[mid_freq_idx]) / np.sum(psd) if np.sum(psd) > 0 else 0
        new_features['high_freq_energy'] = np.sum(psd[high_freq_idx]) / np.sum(psd) if np.sum(psd) > 0 else 0
        
        # Ratio des énergies
        new_features['low_high_ratio'] = (new_features['low_freq_energy'] / new_features['high_freq_energy'] 
                                         if new_features['high_freq_energy'] > 0 else np.nan)
        
        # Moments spectraux
        new_features['spec_skewness'] = np.sum(((freqs-new_features['spec_centroid'])**3)*psd_norm) / (new_features['spec_bandwidth']**3) if new_features['spec_bandwidth'] > 0 else np.nan
        new_features['spec_kurtosis'] = np.sum(((freqs-new_features['spec_centroid'])**4)*psd_norm) / (new_features['spec_bandwidth']**4) if new_features['spec_bandwidth'] > 0 else np.nan
        
        # Flatness spectrale
        new_features['spec_flatness'] = np.exp(np.mean(np.log(psd + 1e-10))) / (np.mean(psd) + 1e-10)
    
    # 3) Pics & saturation
    thr = ts.mean()+2*ts.std()
    peaks, peak_props = signal.find_peaks(ts, height=thr)
    new_features['num_peaks'] = len(peaks)
    new_features['avg_peak_height'] = ts[peaks].mean() if peaks.size else 0
    
    if peaks.size > 1:
        iv = np.diff(peaks)
        new_features['mean_peak_interval'] = iv.mean()
        new_features['std_peak_interval'] = iv.std()
        # Ajout: caractéristiques des pics
        new_features['max_peak_height'] = ts[peaks].max() if peaks.size else 0
        new_features['min_peak_height'] = ts[peaks].min() if peaks.size else 0
        new_features['peak_height_range'] = ts[peaks].max() - ts[peaks].min() if peaks.size else 0
        new_features['peak_height_cv'] = ts[peaks].std() / ts[peaks].mean() if peaks.size and ts[peaks].mean() > 0 else 0
        new_features['min_peak_interval'] = iv.min() if iv.size else 0
        new_features['max_peak_interval'] = iv.max() if iv.size else 0
    else:
        new_features['mean_peak_interval'] = np.nan
        new_features['std_peak_interval'] = np.nan
        new_features['max_peak_height'] = np.nan
        new_features['min_peak_height'] = np.nan
        new_features['peak_height_range'] = np.nan
        new_features['peak_height_cv'] = np.nan
        new_features['min_peak_interval'] = np.nan
        new_features['max_peak_interval'] = np.nan
    
    # Analyse de saturation à différents seuils
    for threshold in [50, 75, 90, 95]:
        mask = ts > threshold
        runs = np.diff(np.r_[0, mask.view(np.int8), 0])
        starts, ends = np.where(runs==1)[0], np.where(runs==-1)[0]
        durs = ends-starts
        
        new_features[f'avg_duration_above_{threshold}'] = durs.mean() if durs.size else 0
        new_features[f'max_duration_above_{threshold}'] = durs.max() if durs.size else 0
        new_features[f'num_episodes_above_{threshold}'] = len(durs)
        new_features[f'pct_time_above_{threshold}'] = np.mean(mask) * 100 if len(mask) > 0 else 0
    
    # 4) Complexité / irrégularité
    try:
        new_features['sample_entropy'] = nolds.sampen(ts, emb_dim=2)
    except:
        new_features['sample_entropy'] = np.nan
    try:
        new_features['fractal_dim_higuchi'] = nolds.higuchi_fd(ts, kmax=10)
    except:
        new_features['fractal_dim_higuchi'] = np.nan
    
    new_features['perm_entropy'] = ant.perm_entropy(ts, order=3, normalize=True)
    
    # Ajout: entropie de permutation à différents ordres
    for order in [2, 4, 5]:
        try:
            new_features[f'perm_entropy_order{order}'] = ant.perm_entropy(ts, order=order, normalize=True)
        except:
            new_features[f'perm_entropy_order{order}'] = np.nan
    
    # Paramètres de Hjorth
    activity = np.var(ts)
    mobility = np.sqrt(np.var(diffs)/activity) if activity>0 else np.nan
    diff2 = np.diff(diffs) if len(diffs) > 1 else np.array([0])
    complexity = (np.sqrt(np.var(diff2)/np.var(diffs))/mobility) if mobility and np.var(diffs)>0 else np.nan
    
    new_features.update({
        'hjorth_activity': activity,
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity
    })
    
    new_features['zero_crossing_rate'] = ((ts[:-1]*ts[1:]<0).sum())/len(ts) if len(ts) > 1 else 0
    
    # Ajout: Approximate Entropy
    try:
        new_features['approx_entropy'] = nolds.app_entropy(ts, emb_dim=2)
    except:
        new_features['approx_entropy'] = np.nan
    
    # Ajout: Correlation Dimension
    try:
        new_features['corr_dim'] = nolds.corr_dim(ts, emb_dim=2)
    except:
        new_features['corr_dim'] = np.nan
    
    # Ajout: Detrended Fluctuation Analysis
    try:
        new_features['dfa'] = nolds.dfa(ts)
    except:
        new_features['dfa'] = np.nan
    
    # 5) Décomposition saisonnière
    try:
        sd = seasonal_decompose(ts, period=24, model='additive', extrapolate_trend='freq')
        new_features['var_trend'] = np.nanvar(sd.trend)
        new_features['var_seasonal'] = np.nanvar(sd.seasonal)
        new_features['var_resid'] = np.nanvar(sd.resid)
        
        # Ajout: statistiques supplémentaires sur les composantes
        new_features['mean_trend'] = np.nanmean(sd.trend)
        new_features['mean_seasonal'] = np.nanmean(sd.seasonal)
        new_features['mean_resid'] = np.nanmean(sd.resid)
        
        new_features['max_trend'] = np.nanmax(sd.trend)
        new_features['min_trend'] = np.nanmin(sd.trend)
        new_features['trend_range'] = np.nanmax(sd.trend) - np.nanmin(sd.trend)
        
        new_features['max_seasonal'] = np.nanmax(sd.seasonal)
        new_features['min_seasonal'] = np.nanmin(sd.seasonal)
        new_features['seasonal_range'] = np.nanmax(sd.seasonal) - np.nanmin(sd.seasonal)
        
        # Ratio des variances
        total_var = new_features['var_trend'] + new_features['var_seasonal'] + new_features['var_resid']
        if total_var > 0:
            new_features['trend_var_ratio'] = new_features['var_trend'] / total_var
            new_features['seasonal_var_ratio'] = new_features['var_seasonal'] / total_var
            new_features['resid_var_ratio'] = new_features['var_resid'] / total_var
    except:
        new_features.update(dict.fromkeys([
            'var_trend', 'var_seasonal', 'var_resid', 
            'mean_trend', 'mean_seasonal', 'mean_resid',
            'max_trend', 'min_trend', 'trend_range',
            'max_seasonal', 'min_seasonal', 'seasonal_range',
            'trend_var_ratio', 'seasonal_var_ratio', 'resid_var_ratio'
        ], np.nan))
    
    # Ajouter la date du jour à group si elle n'existe pas déjà
    group['date'] = group['event_date_time'].dt.date
    
    # Calculer les dépassements par jour
    daily_groups = group.groupby('date')
    daily_exceed_stats = {threshold: [] for threshold in [50, 60, 70, 80, 90, 95]}
    
    for _, day_data in daily_groups:
        day_values = day_data['event_value_norm'].values
        for threshold in [50, 60, 70, 80, 90, 95]:
            daily_exceed_stats[threshold].append(np.sum(day_values > threshold))
    
    # Moyenne des dépassements par jour pour différents seuils
    for threshold in [50, 60, 70, 80, 90, 95]:
        new_features[f'avg_daily_exceed_{threshold}pct'] = np.mean(daily_exceed_stats[threshold]) if daily_exceed_stats[threshold] else 0
        new_features[f'max_daily_exceed_{threshold}pct'] = np.max(daily_exceed_stats[threshold]) if daily_exceed_stats[threshold] else 0
    
    # Calculer les dépassements par semaine
    group['week'] = group['event_date_time'].dt.isocalendar().week
    
    weekly_groups = group.groupby('week')
    weekly_exceed_stats = {threshold: [] for threshold in [50, 60, 70, 80, 90, 95]}
    
    for _, week_data in weekly_groups:
        week_values = week_data['event_value_norm'].values
        for threshold in [50, 60, 70, 80, 90, 95]:
            weekly_exceed_stats[threshold].append(np.sum(week_values > threshold))
    
    # Moyenne des dépassements par semaine pour différents seuils
    for threshold in [50, 60, 70, 80, 90, 95]:
        new_features[f'avg_weekly_exceed_{threshold}pct'] = np.mean(weekly_exceed_stats[threshold]) if weekly_exceed_stats[threshold] else 0
        new_features[f'max_weekly_exceed_{threshold}pct'] = np.max(weekly_exceed_stats[threshold]) if weekly_exceed_stats[threshold] else 0
    
    # Ajout: Analyse par heure de la journée
    group['hour'] = group['event_date_time'].dt.hour
    hourly_stats = {}
    
    for hour in range(24):
        hour_data = group[group['hour'] == hour]['event_value_norm'].values
        if len(hour_data) > 0:
            hourly_stats[f'hour_{hour}_mean'] = np.mean(hour_data)
            hourly_stats[f'hour_{hour}_std'] = np.std(hour_data)
            hourly_stats[f'hour_{hour}_max'] = np.max(hour_data)
            hourly_stats[f'hour_{hour}_min'] = np.min(hour_data)
    
    # Ajout: Statistiques jour vs nuit
    day_mask = (group['hour'] >= 8) & (group['hour'] < 20)
    day_values = group.loc[day_mask, 'event_value_norm'].values
    night_values = group.loc[~day_mask, 'event_value_norm'].values
    
    if len(day_values) > 0 and len(night_values) > 0:
        new_features['day_mean'] = np.mean(day_values)
        new_features['night_mean'] = np.mean(night_values)
        new_features['day_std'] = np.std(day_values)
        new_features['night_std'] = np.std(night_values)
        new_features['day_night_ratio'] = new_features['day_mean'] / new_features['night_mean'] if new_features['night_mean'] > 0 else np.nan
        new_features['day_night_diff'] = new_features['day_mean'] - new_features['night_mean']
    
    # Ajout: Statistiques jours ouvrables vs week-end
    group['is_weekend'] = group['event_date_time'].dt.dayofweek >= 5
    weekday_values = group.loc[~group['is_weekend'], 'event_value_norm'].values
    weekend_values = group.loc[group['is_weekend'], 'event_value_norm'].values
    
    if len(weekday_values) > 0 and len(weekend_values) > 0:
        new_features['weekday_mean'] = np.mean(weekday_values)
        new_features['weekend_mean'] = np.mean(weekend_values)
        new_features['weekday_std'] = np.std(weekday_values)
        new_features['weekend_std'] = np.std(weekend_values)
        new_features['weekday_weekend_ratio'] = new_features['weekday_mean'] / new_features['weekend_mean'] if new_features['weekend_mean'] > 0 else np.nan
        new_features['weekday_weekend_diff'] = new_features['weekday_mean'] - new_features['weekend_mean']
    
    # Caractéristiques catch22
    catch22_dict = {}
    try:
        # Utiliser l'API correcte de pycatch22
        catch22_features = pycatch22.catch22_all(ts)
        catch22_dict = {name: value for name, value in zip(catch22_features['names'], catch22_features['values'])}
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques catch22: {e}")
        # Créer un dictionnaire avec des valeurs NaN en cas d'erreur
        feature_names = [
            'DN_HistogramMode_5', 'DN_HistogramMode_10', 'SB_BinaryStats_mean_longstretch1',
            'DN_OutlierInclude_p_001_mdrmd', 'DN_OutlierInclude_n_001_mdrmd', 'CO_f1ecac',
            'CO_FirstMin_ac', 'SP_Summaries_welch_rect_area_5_1', 'SP_Summaries_welch_rect_centroid',
            'FC_LocalSimple_mean3_stderr', 'CO_trev_1_num', 'CO_HistogramAMI_even_2_5',
            'IN_AutoMutualInfoStats_40_gaussian_fmmi', 'MD_hrv_classic_pnn40', 'SB_BinaryStats_diff_longstretch0',
            'SB_MotifThree_quantile_hh', 'FC_LocalSimple_mean1_tauresrat', 'CO_Embed2_Dist_tau_d_expfit_meandiff',
            'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1', 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
            'SB_TransitionMatrix_3ac_sumdiagcov', 'PD_PeriodicityWang_th0_01'
        ]
        catch22_dict = {name: np.nan for name in feature_names}
    
    # Statistiques descriptives de base
    basic_stats = {
        'mean': np.mean(ts),
        'std': np.std(ts),
        'min': np.min(ts),
        'max': np.max(ts),
        'median': np.median(ts),
        'iqr': np.percentile(ts, 75) - np.percentile(ts, 25),
        'skewness': pd.Series(ts).skew(),
        'kurtosis': pd.Series(ts).kurtosis(),
        # Ajout: statistiques supplémentaires
        'range': np.max(ts) - np.min(ts),
        'cv': np.std(ts) / np.mean(ts) if np.mean(ts) > 0 else np.nan,
        'mad': np.median(np.abs(ts - np.median(ts))),  # Median Absolute Deviation
        'energy': np.sum(ts**2),
        'rms': np.sqrt(np.mean(ts**2)),  # Root Mean Square
        'crest_factor': np.max(np.abs(ts)) / np.sqrt(np.mean(ts**2)) if np.mean(ts**2) > 0 else np.nan
    }
    
    # Percentiles pour inavgload/outavgload
    percentiles = {
        'load_10p': np.percentile(ts, 10),
        'load_25p': np.percentile(ts, 25),
        'load_50p': np.percentile(ts, 50),
        'load_75p': np.percentile(ts, 75),
        'load_90p': np.percentile(ts, 90),
        'load_95p': np.percentile(ts, 95),
        'load_99p': np.percentile(ts, 99)
    }
    
    # Asymétrie à différents percentiles
    asym_percentiles = {
        'asym_1p': np.percentile(ts, 1),
        'asym_5p': np.percentile(ts, 5),
        'asym_25p': np.percentile(ts, 25),
        'asym_50p': np.percentile(ts, 50),
        'asym_75p': np.percentile(ts, 75),
        'asym_95p': np.percentile(ts, 95),
        'asym_99p': np.percentile(ts, 99)
    }
    
    # Ajout: Ratios entre percentiles
    percentile_ratios = {
        'ratio_75_25': percentiles['load_75p'] / percentiles['load_25p'] if percentiles['load_25p'] > 0 else np.nan,
        'ratio_90_10': percentiles['load_90p'] / percentiles['load_10p'] if percentiles['load_10p'] > 0 else np.nan,
        'ratio_95_5': percentiles['load_95p'] / asym_percentiles['asym_5p'] if asym_percentiles['asym_5p'] > 0 else np.nan,
        'ratio_99_1': percentiles['load_99p'] / asym_percentiles['asym_1p'] if asym_percentiles['asym_1p'] > 0 else np.nan
    }
    
    # Statistiques hebdomadaires
    weekly_stats = {}
    # Grouper par semaine
    weekly_groups = group.groupby('week')
    
    for i, (week, week_data) in enumerate(weekly_groups):
        week_ts = week_data['event_value'].values
        if len(week_ts) > 0:
            weekly_stats[f'week_{i}_mean'] = np.mean(week_ts)
            weekly_stats[f'week_{i}_min'] = np.min(week_ts)
            weekly_stats[f'week_{i}_max'] = np.max(week_ts)
            weekly_stats[f'week_{i}_std'] = np.std(week_ts)
            weekly_stats[f'week_{i}_median'] = np.median(week_ts)
    
    # Statistiques par jour de la semaine
    dow_stats = {}
    group['day_of_week'] = group['event_date_time'].dt.dayofweek
    dow_groups = group.groupby('day_of_week')
    
    for dow, dow_data in dow_groups:
        dow_ts = dow_data['event_value'].values
        if len(dow_ts) > 0:
            dow_stats[f'dow_{dow}_mean'] = np.mean(dow_ts)
            dow_stats[f'dow_{dow}_std'] = np.std(dow_ts)
            dow_stats[f'dow_{dow}_max'] = np.max(dow_ts)
            dow_stats[f'dow_{dow}_min'] = np.min(dow_ts)
            dow_stats[f'dow_{dow}_range'] = np.max(dow_ts) - np.min(dow_ts)
    
    # Statistiques par heure
    hour_stats = {}
    group['event_hour'] = group['event_date_time'].dt.hour
    hour_groups = group.groupby('event_hour')
    
    for hour, hour_data in hour_groups:
        hour_ts = hour_data['event_value'].values
        if len(hour_ts) > 0:
            hour_stats[f'hour_{hour}_mean'] = np.mean(hour_ts)
            hour_stats[f'hour_{hour}_std'] = np.std(hour_ts)
            hour_stats[f'hour_{hour}_max'] = np.max(hour_ts)
            hour_stats[f'hour_{hour}_min'] = np.min(hour_ts)
    
    # Tendance et saisonnalité
    if len(ts) > 1:
        # Tendance linéaire simple
        x = np.arange(len(ts))
        trend_coeffs = np.polyfit(x, ts, 1)
        trend_slope = trend_coeffs[0]  # Pente de la tendance
        trend_intercept = trend_coeffs[1]  # Ordonnée à l'origine
        
        # Tendance quadratique
        if len(ts) > 2:
            quad_coeffs = np.polyfit(x, ts, 2)
            quad_a = quad_coeffs[0]  # Coefficient quadratique
        else:
            quad_a = np.nan
        
        # Autocorrélation à différents lags
        acf_1 = pd.Series(ts).autocorr(lag=1)
        acf_7 = pd.Series(ts).autocorr(lag=7) if len(ts) > 7 else np.nan
        acf_24 = pd.Series(ts).autocorr(lag=24) if len(ts) > 24 else np.nan
        
        # Ajout: autocorrélations supplémentaires
        acf_2 = pd.Series(ts).autocorr(lag=2) if len(ts) > 2 else np.nan
        acf_3 = pd.Series(ts).autocorr(lag=3) if len(ts) > 3 else np.nan
        acf_12 = pd.Series(ts).autocorr(lag=12) if len(ts) > 12 else np.nan
        
        trend_seasonality = {
            'trend_slope': trend_slope,
            'trend_intercept': trend_intercept,
            'quad_coeff': quad_a,
            'acf_lag1': acf_1,
            'acf_lag2': acf_2,
            'acf_lag3': acf_3,
            'acf_lag7': acf_7,
            'acf_lag12': acf_12,
            'acf_lag24': acf_24
        }
    else:
        trend_seasonality = {
            'trend_slope': np.nan,
            'trend_intercept': np.nan,
            'quad_coeff': np.nan,
            'acf_lag1': np.nan,
            'acf_lag2': np.nan,
            'acf_lag3': np.nan,
            'acf_lag7': np.nan,
            'acf_lag12': np.nan,
            'acf_lag24': np.nan
        }
    
    # Caractéristiques statiques (prendre la première valeur car elles ne changent pas avec le temps)
    static_features = {
        'device_role': group['device_role'].iloc[0],
        'site_longitude': group['site_longitude'].iloc[0],
        'site_latitude': group['site_latitude'].iloc[0],
        'site_criticality': group['site_criticality'].iloc[0],
        'site_country': group['site_country'].iloc[0],
        'Criticality_unknown': group['Criticality_unknown'].iloc[0] if 'Criticality_unknown' in group.columns else np.nan,
    }
    
    # Combiner toutes les caractéristiques
    all_features = {
        'ci_name': group['ci_name'].iloc[0],
        **static_features,
        **catch22_dict,
        **basic_stats,
        **percentiles,
        **asym_percentiles,
        **percentile_ratios,
        **new_features,
        **weekly_stats,
        **dow_stats,
        **hour_stats,
        **hourly_stats,
        **trend_seasonality
    }
    
    return all_features

# Extraire les caractéristiques pour chaque équipement
print("Extraction des caractéristiques des séries temporelles...")
# Utiliser une liste pour collecter les résultats
features_list = []
for name, group in tqdm(df.groupby('ci_name')):
    features = extract_time_series_features(group)
    features_list.append(features)

# Créer un DataFrame à partir de la liste de dictionnaires
features_df = pd.DataFrame(features_list)

# Enregistrer le features_df
features_df.to_csv("features_df.csv")

# Vérifier les valeurs manquantes dans les caractéristiques extraites
print("\nValeurs manquantes dans les caractéristiques extraites:")
missing_values = features_df.isnull().sum()
print(missing_values[missing_values > 0])

# Imputer les valeurs manquantes
print("Imputation des valeurs manquantes...")
for col in features_df.columns:
    if features_df[col].isnull().sum() > 0:
        if pd.api.types.is_numeric_dtype(features_df[col]):
            features_df[col] = features_df[col].fillna(features_df[col].median())
        else:
            features_df[col] = features_df[col].fillna(features_df[col].mode()[0])

# Séparer les caractéristiques catégorielles et numériques
categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_cols.remove('ci_name')  # Exclure l'identifiant
numerical_cols = features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCaractéristiques catégorielles: {categorical_cols}")
print(f"Caractéristiques numériques: {numerical_cols}")

# Détection et suppression des valeurs aberrantes
print("Détection des valeurs aberrantes...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(features_df[numerical_cols].fillna(0))
print(f"Nombre de valeurs aberrantes détectées: {np.sum(outliers == -1)}")
print(f"Pourcentage de valeurs aberrantes: {np.sum(outliers == -1) / len(outliers) * 100:.2f}%")

# Créer un DataFrame sans les valeurs aberrantes
features_df_no_outliers = features_df[outliers == 1].copy()
print(f"Forme du DataFrame après suppression des valeurs aberrantes: {features_df_no_outliers.shape}")

# Sélection de caractéristiques basée sur la variance
print("Sélection de caractéristiques basée sur la variance...")
selector = VarianceThreshold(threshold=0.01)  # Supprimer les caractéristiques avec une variance < 0.01
X_var_selected = selector.fit_transform(features_df_no_outliers[numerical_cols])
selected_features = [numerical_cols[i] for i in range(len(numerical_cols)) if selector.get_support()[i]]
print(f"Nombre de caractéristiques après sélection par variance: {len(selected_features)}")

# Analyse en Composantes Principales (PCA)
print("Application de PCA...")
pca = PCA(n_components=0.95)  # Conserver 95% de la variance
X_pca = pca.fit_transform(StandardScaler().fit_transform(features_df_no_outliers[selected_features]))
print(f"Nombre de composantes PCA retenues: {pca.n_components_}")
print(f"Variance expliquée: {np.sum(pca.explained_variance_ratio_):.2f}")

# Visualiser la variance expliquée par composante
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance expliquée cumulée')
plt.grid(True)
plt.savefig('results_catch22/Interface/dimensionality_reduction/pca_variance_explained.png', dpi=300)
plt.close()

# Visualisation t-SNE des données
print("Application de t-SNE pour visualisation...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(StandardScaler().fit_transform(features_df_no_outliers[selected_features]))

plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
plt.title('Visualisation t-SNE des données')
plt.savefig('results_catch22/Interface/dimensionality_reduction/tsne_visualization.png', dpi=300)
plt.close()

# Prétraitement des données pour le clustering
print("Prétraitement des données...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), selected_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Appliquer le prétraitement
X = preprocessor.fit_transform(features_df_no_outliers[selected_features + categorical_cols])

# Conserver les noms des équipements pour l'interprétation
ci_names = features_df_no_outliers['ci_name'].values

print(f"Forme des données après prétraitement: {X.shape}")

# Fonction pour évaluer les clusters
def evaluate_clustering(X, labels):
    counts = np.bincount(labels[labels >= 0])  # ignore bruit si DBSCAN

    if len(np.unique(labels)) <= 1:
        return {'silhouette': -1, 'calinski_harabasz': -1, 'davies_bouldin': -1}
    
    try:
        silhouette = silhouette_score(X, labels)
    except:
        silhouette = -1
    
    try:
        calinski_harabasz = calinski_harabasz_score(X, labels)
    except:
        calinski_harabasz = -1
    
    try:
        davies_bouldin = davies_bouldin_score(X, labels)
    except:
        davies_bouldin = -1
    
    return {
        'silhouette': silhouette,
        'calinski_harabasz': calinski_harabasz,
        'davies_bouldin': davies_bouldin,
        'counts': counts
    }

# Fonction pour visualiser les clusters avec PCA et t-SNE
def visualize_clusters(X, labels, algorithm_name, n_clusters=None, use_tsne=False):
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    if use_tsne:
        # Réduire à 2 dimensions avec t-SNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        X_reduced = reducer.fit_transform(X_imputed)
        method = 't-SNE'
    else:
        # Réduire à 2 dimensions avec PCA
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X_imputed)
        method = 'PCA'
    
    # Créer le graphique
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    
    # Ajouter les informations sur le graphique
    if n_clusters:
        plt.title(f'Visualisation {method} des clusters - {algorithm_name} (k={n_clusters})')
    else:
        plt.title(f'Visualisation {method} des clusters - {algorithm_name}')
    plt.xlabel(f'Composante 1')
    plt.ylabel(f'Composante 2')
    
    # Sauvegarder le graphique
    if n_clusters:
        plt.savefig(f'results_catch22/Interface/{method.lower()}_clusters_{algorithm_name}_k{n_clusters}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'results_catch22/Interface/{method.lower()}_clusters_{algorithm_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Fonction pour analyser l'importance des caractéristiques dans chaque cluster
def analyze_feature_importance(features_df, labels, selected_features, output_dir):
    features_with_clusters = features_df.copy()
    features_with_clusters['cluster'] = labels
    
    # Créer un dossier pour les résultats
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistiques par cluster
    cluster_stats = features_with_clusters.groupby('cluster')[selected_features].mean()
    cluster_stats.to_csv(f'{output_dir}/cluster_feature_means.csv')
    
    # Visualiser les caractéristiques les plus discriminantes
    plt.figure(figsize=(15, 10))
    
    # Calculer l'importance des caractéristiques (variance entre clusters)
    feature_importance = {}
    for feature in selected_features:
        # Calculer la variance entre les moyennes des clusters
        cluster_means = features_with_clusters.groupby('cluster')[feature].mean()
        feature_importance[feature] = cluster_means.var()
    
    # Trier les caractéristiques par importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:10]]  # Top 10 caractéristiques
    
    # Créer un boxplot pour chaque caractéristique importante
    for i, feature in enumerate(top_features):
        plt.subplot(2, 5, i+1)
        sns.boxplot(x='cluster', y=feature, data=features_with_clusters)
        plt.title(feature)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_features_boxplot.png', dpi=300)
    plt.close()
    
    # Créer un heatmap des moyennes des caractéristiques par cluster
    plt.figure(figsize=(15, 10))
    sns.heatmap(cluster_stats[top_features], annot=True, cmap='viridis', fmt='.2f')
    plt.title('Moyennes des caractéristiques importantes par cluster')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_heatmap.png', dpi=300)
    plt.close()
    
    return top_features

# Fonction pour exécuter et évaluer différents algorithmes de clustering
def run_clustering_algorithms(X, ci_names, selected_features, features_df_no_outliers):
    results = []
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Essayer différentes méthodes de réduction de dimensionnalité
    print("Application de PCA pour le clustering...")
    pca = PCA(n_components=min(20, X_imputed.shape[1]))
    X_pca = pca.fit_transform(X_imputed)
    print(f"Variance expliquée avec {pca.n_components_} composantes: {np.sum(pca.explained_variance_ratio_):.2f}")
    
    # Algorithmes avec nombre de clusters fixe
    for k in range(2, 11):
        print(f"\nTest avec {k} clusters...")
        
        # KMeans avec différentes initialisations
        print("Exécution de KMeans...")
        best_kmeans_score = -1
        best_kmeans_labels = None
        
        for init in ['k-means++', 'random']:
            for n_init in [10, 20]:
                kmeans = KMeans(n_clusters=k, init=init, n_init=n_init, random_state=42)
                kmeans_labels = kmeans.fit_predict(X_imputed)
                score = silhouette_score(X_imputed, kmeans_labels) if len(np.unique(kmeans_labels)) > 1 else -1
                
                if score > best_kmeans_score:
                    best_kmeans_score = score
                    best_kmeans_labels = kmeans_labels
        
        kmeans_metrics = evaluate_clustering(X_imputed, best_kmeans_labels)
        results.append({
            'algorithm': 'KMeans',
            'n_clusters': k,
            'labels': best_kmeans_labels,
            **kmeans_metrics
        })
        visualize_clusters(X_imputed, best_kmeans_labels, 'KMeans', k)
        visualize_clusters(X_imputed, best_kmeans_labels, 'KMeans', k, use_tsne=True)
        
        # KMeans sur données PCA
        kmeans_pca = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_pca_labels = kmeans_pca.fit_predict(X_pca)
        kmeans_pca_metrics = evaluate_clustering(X_pca, kmeans_pca_labels)
        results.append({
            'algorithm': 'KMeans_PCA',
            'n_clusters': k,
            'labels': kmeans_pca_labels,
            **kmeans_pca_metrics
        })
        visualize_clusters(X_pca, kmeans_pca_labels, 'KMeans_PCA', k)
        
        # Gaussian Mixture Model
        print("Exécution de GMM...")
        for cov_type in ['full', 'tied', 'diag']:
            gmm = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=42)
            gmm_labels = gmm.fit_predict(X_imputed)
            gmm_metrics = evaluate_clustering(X_imputed, gmm_labels)
            results.append({
                'algorithm': f'GMM_{cov_type}',
                'n_clusters': k,
                'labels': gmm_labels,
                **gmm_metrics
            })
            visualize_clusters(X_imputed, gmm_labels, f'GMM_{cov_type}', k)
        
        # KMedoids
        print("Exécution de KMedoids...")
        try:
            kmedoids = KMedoids(n_clusters=k, random_state=42, metric='euclidean', method='pam', init='k-medoids++')
            kmedoids_labels = kmedoids.fit_predict(X_imputed)
            kmedoids_metrics = evaluate_clustering(X_imputed, kmedoids_labels)
            results.append({
                'algorithm': 'KMedoids',
                'n_clusters': k,
                'labels': kmedoids_labels,
                **kmedoids_metrics
            })
            visualize_clusters(X_imputed, kmedoids_labels, 'KMedoids', k)
        except Exception as e:
            print(f"Erreur avec KMedoids: {e}")
        
        # Birch
        print("Exécution de Birch...")
        birch = Birch(n_clusters=k)
        birch_labels = birch.fit_predict(X_imputed)
        birch_metrics = evaluate_clustering(X_imputed, birch_labels)
        results.append({
            'algorithm': 'Birch',
            'n_clusters': k,
            'labels': birch_labels,
            **birch_metrics
        })
        visualize_clusters(X_imputed, birch_labels, 'Birch', k)
        
        # Hierarchical Clustering avec différentes métriques
        print("Exécution de Hierarchical Clustering...")
        for linkage_method in ['ward', 'complete', 'average']:
            try:
                hclust = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
                hclust_labels = hclust.fit_predict(X_imputed)
                hclust_metrics = evaluate_clustering(X_imputed, hclust_labels)
                results.append({
                    'algorithm': f'HClust_{linkage_method}',
                    'n_clusters': k,
                    'labels': hclust_labels,
                    **hclust_metrics
                })
                visualize_clusters(X_imputed, hclust_labels, f'HClust_{linkage_method}', k)
            except Exception as e:
                print(f"Erreur avec Hierarchical Clustering ({linkage_method}): {e}")
    
    # Algorithmes qui déterminent automatiquement le nombre de clusters
    
    # DBSCAN avec différents paramètres
    print("\nExécution de DBSCAN...")
    for eps in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        for min_samples in [5, 10, 15]:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                dbscan_labels = dbscan.fit_predict(X_imputed)
                
                # Vérifier si DBSCAN a trouvé plus d'un cluster (hors bruit)
                n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                if n_clusters > 1:
                    dbscan_metrics = evaluate_clustering(X_imputed, dbscan_labels)
                    results.append({
                        'algorithm': f'DBSCAN_eps{eps}_ms{min_samples}',
                        'n_clusters': n_clusters,
                        'labels': dbscan_labels,
                        **dbscan_metrics
                    })
                    visualize_clusters(X_imputed, dbscan_labels, f'DBSCAN_eps{eps}_ms{min_samples}')
                    print(f"DBSCAN (eps={eps}, min_samples={min_samples}) a trouvé {n_clusters} clusters")
                else:
                    print(f"DBSCAN (eps={eps}, min_samples={min_samples}) n'a pas trouvé de clusters significatifs")
            except Exception as e:
                print(f"Erreur avec DBSCAN (eps={eps}, min_samples={min_samples}): {e}")
    
    # Analyser les résultats et sélectionner le meilleur algorithme
    best_silhouette = -1
    best_result = None
    
    for result in results:
        if result['silhouette'] > best_silhouette:
            best_silhouette = result['silhouette']
            best_result = result
    
    if best_result:
        print(f"\nMeilleur algorithme: {best_result['algorithm']} avec {best_result['n_clusters']} clusters")
        print(f"Score de silhouette: {best_result['silhouette']}")
        
        # Analyser l'importance des caractéristiques pour le meilleur clustering
        top_features = analyze_feature_importance(
            features_df_no_outliers, 
            best_result['labels'], 
            selected_features,
                        f"results_catch22/Interface/feature_analysis/{best_result['algorithm']}_k{best_result['n_clusters']}"
        )
        
        # Créer un DataFrame avec les noms des équipements et leurs clusters
        cluster_assignments = pd.DataFrame({
            'ci_name': ci_names,
            'cluster': best_result['labels']
        })
        
        # Sauvegarder les assignations de clusters
        cluster_assignments.to_csv(f'results_catch22/Interface/best_cluster_assignments_{best_result["algorithm"]}_k{best_result["n_clusters"]}.csv', index=False)
    
    return results

# Exécuter les algorithmes de clustering
print("\nExécution des algorithmes de clustering...")
clustering_results = run_clustering_algorithms(X, ci_names, selected_features, features_df_no_outliers)

# Créer un DataFrame avec les résultats
results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'labels'} for r in clustering_results])

# Sauvegarder les résultats dans un CSV
results_df.to_csv('results_catch22/Interface/clustering_metrics.csv', index=False)
print("\nRésultats sauvegardés dans 'results_catch22/Interface/clustering_metrics.csv'")

# Afficher les résultats
print("\nRésultats des métriques de clustering:")
print(results_df)

# Identifier le meilleur algorithme selon chaque métrique
best_silhouette = results_df.loc[results_df['silhouette'].idxmax()]
best_calinski = results_df.loc[results_df['calinski_harabasz'].idxmax()]
best_davies = results_df.loc[results_df['davies_bouldin'].idxmin()]  # Pour Davies-Bouldin, plus petit = meilleur

print("\nMeilleur algorithme selon le score de silhouette:")
print(best_silhouette)
print("\nMeilleur algorithme selon le score de Calinski-Harabasz:")
print(best_calinski)
print("\nMeilleur algorithme selon le score de Davies-Bouldin:")
print(best_davies)

# Visualiser les métriques pour les algorithmes avec nombre de clusters fixe
fixed_k_algorithms = ['KMeans', 'KMeans_PCA', 'GMM_full', 'KMedoids', 'Birch', 'HClust_ward']
fixed_k_results = results_df[results_df['algorithm'].isin(fixed_k_algorithms)]

# Créer des graphiques pour chaque métrique
plt.figure(figsize=(15, 5))

# Silhouette Score (plus élevé = meilleur)
plt.subplot(1, 3, 1)
for algo in fixed_k_algorithms:
    algo_results = fixed_k_results[fixed_k_results['algorithm'] == algo]
    if not algo_results.empty:
        plt.plot(algo_results['n_clusters'], algo_results['silhouette'], marker='o', label=algo)
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de Silhouette')
plt.title('Score de Silhouette par algorithme')
plt.legend()
plt.grid(True)

# Calinski-Harabasz Score (plus élevé = meilleur)
plt.subplot(1, 3, 2)
for algo in fixed_k_algorithms:
    algo_results = fixed_k_results[fixed_k_results['algorithm'] == algo]
    if not algo_results.empty:
        plt.plot(algo_results['n_clusters'], algo_results['calinski_harabasz'], marker='o', label=algo)
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de Calinski-Harabasz')
plt.title('Score de Calinski-Harabasz par algorithme')
plt.legend()
plt.grid(True)

# Davies-Bouldin Score (plus bas = meilleur)
plt.subplot(1, 3, 3)
for algo in fixed_k_algorithms:
    algo_results = fixed_k_results[fixed_k_results['algorithm'] == algo]
    if not algo_results.empty:
        plt.plot(algo_results['n_clusters'], algo_results['davies_bouldin'], marker='o', label=algo)
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de Davies-Bouldin')
plt.title('Score de Davies-Bouldin par algorithme')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results_catch22/Interface/clustering_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Identifier la meilleure configuration globale
# On peut utiliser une approche de classement pour combiner les métriques
results_df['silhouette_rank'] = results_df['silhouette'].rank(ascending=False)
results_df['calinski_rank'] = results_df['calinski_harabasz'].rank(ascending=False)
results_df['davies_rank'] = results_df['davies_bouldin'].rank(ascending=True)  # Plus petit = meilleur
results_df['combined_rank'] = results_df['silhouette_rank'] + results_df['calinski_rank'] + results_df['davies_rank']

best_overall = results_df.loc[results_df['combined_rank'].idxmin()]

print("\nMeilleure configuration globale (basée sur le classement combiné des métriques):")
print(best_overall)

# Sauvegarder le DataFrame avec les rangs
results_df.to_csv('results_catch22/Interface/clustering_metrics_with_ranks.csv', index=False)

# Récupérer les labels du meilleur algorithme
best_algo = best_overall['algorithm']
best_n_clusters = best_overall['n_clusters']
best_result = next((r for r in clustering_results if r['algorithm'] == best_algo and r['n_clusters'] == best_n_clusters), None)

if best_result:
    best_labels = best_result['labels']
    
    # Créer un DataFrame avec les noms des équipements et leurs clusters
    cluster_assignments = pd.DataFrame({
        'ci_name': ci_names,
        'cluster': best_labels
    })
    
    # Sauvegarder les assignations de clusters
    cluster_assignments.to_csv('results_catch22/Interface/best_cluster_assignments.csv', index=False)
    
    # Visualiser le meilleur clustering avec PCA et t-SNE
    visualize_clusters(X, best_labels, f"{best_algo}_best", best_n_clusters)
    visualize_clusters(X, best_labels, f"{best_algo}_best", best_n_clusters, use_tsne=True)
    
    # Analyser les caractéristiques de chaque cluster
    features_with_clusters = features_df_no_outliers.copy()
    features_with_clusters['cluster'] = best_labels
    
    # Statistiques par cluster
    numeric_cols = features_with_clusters.select_dtypes(include=['int64','float64']).columns
    cluster_stats = features_with_clusters.groupby('cluster')[numeric_cols].mean()
    cluster_stats.to_csv('results_catch22/Interface/cluster_statistics.csv')
    
    # Analyse détaillée des caractéristiques par cluster
    analyze_feature_importance(
        features_df_no_outliers, 
        best_labels, 
        selected_features,
        f"results_catch22/Interface/feature_analysis/best_overall"
    )
    
    # Visualiser la distribution des clusters
    plt.figure(figsize=(10, 6))
    cluster_counts = pd.Series(best_labels).value_counts().sort_index()
    cluster_counts.plot(kind='bar')
    plt.title(f'Distribution des clusters - {best_algo} (k={best_n_clusters})')
    plt.xlabel('Cluster')
    plt.ylabel('Nombre d\'équipements')
    plt.grid(True, axis='y')
    plt.savefig('results_catch22/Interface/cluster_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Créer un dendrogramme pour le clustering hiérarchique si c'est le meilleur
    if 'HClust' in best_algo:
        plt.figure(figsize=(15, 10))
        Z = linkage(X, method=best_algo.split('_')[1] if '_' in best_algo else 'ward')
        dendrogram(Z, truncate_mode='lastp', p=best_n_clusters*2, leaf_rotation=90.)
        plt.title(f'Dendrogramme du clustering hiérarchique ({best_algo})')
        plt.xlabel('Équipements')
        plt.ylabel('Distance')
        plt.savefig('results_catch22/Interface/hierarchical_dendrogram.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\nAnalyse de clustering terminée avec succès!")
    print(f"Meilleur algorithme: {best_algo} avec {best_n_clusters} clusters")
    print(f"Score de silhouette: {best_overall['silhouette']}")
    print(f"Score de Calinski-Harabasz: {best_overall['calinski_harabasz']}")
    print(f"Score de Davies-Bouldin: {best_overall['davies_bouldin']}")
else:
    print("\nErreur: Impossible de récupérer les labels du meilleur algorithme.")

print("\nTous les résultats ont été sauvegardés dans le dossier 'results_catch22/Interface/'.")
