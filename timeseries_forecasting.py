import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import optuna
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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

# Création du dossier pour sauvegarder les résultats
os.makedirs("TimeSeries_forecasting", exist_ok=True)

# 1. Récupération des données
def load_data():
    print("Chargement des données...")
    # Chargez votre jeu de données principal
    df = pd.read_csv('Data\AllMonths\data_cleaned_interface.csv')
    
    # Chargez les assignations de cluster
    cluster_df = pd.read_csv('cluster_assignement.csv')
    
    # Essayez de charger les features d'extraction si disponibles
    try:
        features_df = pd.read_csv('results_catch22\Interface_4\features_df.csv')
        print("Features d'extraction chargées avec succès.")
        return df, cluster_df, features_df
    except:
        print("Features d'extraction non disponibles. Poursuite sans ces features.")
        return df, cluster_df, None

# Fonction pour convertir les colonnes de date
def convert_date_columns(df):
    date_columns = ['event_date_time', 'Date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df

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
            df[col] = imputer.fit_transform(df[[col]])
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

# Fonction pour le feature engineering
def feature_engineering(df, cluster_id):
    print(f"Feature engineering pour le cluster {cluster_id}...")
    
    # Copie pour éviter les warnings
    df = df.copy()
    
    # 1. Features temporelles supplémentaires
    if 'event_date_time' in df.columns:
        # Extraire l'année, le mois, le trimestre
        df['Year'] = df['event_date_time'].dt.year
        df['Month'] = df['event_date_time'].dt.month
        df['Quarter'] = df['event_date_time'].dt.quarter
        
        # Jour de l'année et semaine de l'année
        df['Day_of_year'] = df['event_date_time'].dt.dayofyear
        df['Week_of_year'] = df['event_date_time'].dt.isocalendar().week
        
        # Transformations cycliques pour le mois
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Indicateur de début, milieu et fin de mois
        df['Is_month_start'] = (df['event_date_time'].dt.day <= 5).astype(int)
        df['Is_month_middle'] = ((df['event_date_time'].dt.day > 5) & 
                                (df['event_date_time'].dt.day <= 25)).astype(int)
        df['Is_month_end'] = (df['event_date_time'].dt.day > 25).astype(int)
        
        # Indicateur de début et fin de semaine
        df['Is_week_start'] = (df['event_date_time'].dt.dayofweek == 0).astype(int)
        df['Is_week_end'] = (df['event_date_time'].dt.dayofweek == 4).astype(int)
        
        # Partie de la journée
        df['Part_of_day'] = pd.cut(
            df['Hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['Night', 'Morning', 'Afternoon', 'Evening']
        )
        
        # One-hot encoding pour Part_of_day
        part_of_day_dummies = pd.get_dummies(df['Part_of_day'], prefix='Part_of_day')
        df = pd.concat([df, part_of_day_dummies], axis=1)
        
    # 2. Features de lag supplémentaires (par équipement)
    if 'event_value' in df.columns and 'ci_name' in df.columns:
        # Trier par équipement et date
        df = df.sort_values(['ci_name', 'event_date_time'])
        
        # Créer des lags supplémentaires (2, 3, 6, 12, 24 heures)
        for lag in [2, 3, 6, 12, 24]:
            df[f'Lag_{lag}'] = df.groupby('ci_name')['event_value'].shift(lag)
        
        # Différences entre les lags
        df['Diff_1'] = df['event_value'] - df['Lag_1']
        df['Diff_2'] = df['Lag_1'] - df['Lag_2']
        
        # Taux de changement
        df['Rate_of_change_1'] = df['Diff_1'] / (df['Lag_1'] + 1e-10)  # Éviter division par zéro
        
        # Moyennes mobiles supplémentaires
        for window in [6, 12, 24]:
            df[f'Rolling_mean_{window}'] = df.groupby('ci_name')['event_value'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'Rolling_std_{window}'] = df.groupby('ci_name')['event_value'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std())
            df[f'Rolling_min_{window}'] = df.groupby('ci_name')['event_value'].transform(
                lambda x: x.rolling(window=window, min_periods=1).min())
            df[f'Rolling_max_{window}'] = df.groupby('ci_name')['event_value'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max())
        
        # Écart par rapport à la moyenne mobile
        df['Deviation_from_mean_3'] = df['event_value'] - df['Rolling_mean_3']
        df['Deviation_from_mean_12'] = df['event_value'] - df['Rolling_mean_12']
        
        # Z-score par rapport à la fenêtre mobile
        df['Z_score_12'] = df['Deviation_from_mean_12'] / (df['Rolling_std_12'] + 1e-10)
        
        # Indicateurs de tendance
        df['Trend_up'] = ((df['Rolling_mean_3'] > df['Rolling_mean_12']) & 
                          (df['Rolling_mean_12'] > df['Rolling_mean_24'])).astype(int)
        df['Trend_down'] = ((df['Rolling_mean_3'] < df['Rolling_mean_12']) & 
                           (df['Rolling_mean_12'] < df['Rolling_mean_24'])).astype(int)
        
        # Volatilité relative
        df['Relative_volatility'] = df['Rolling_std_12'] / (df['Rolling_mean_12'] + 1e-10)
    
    # 3. Features basées sur les heures d'ouverture
    if 'site_business_hour' in df.columns and 'Hour' in df.columns:
        # Créer un indicateur si l'heure actuelle est dans les heures d'ouverture
        # Supposons que site_business_hour contient des plages comme "9-17"
        try:
            df['Is_business_hour'] = df.apply(
                lambda row: 1 if row['Hour'] >= int(row['site_business_hour'].split('-')[0]) and 
                                 row['Hour'] <= int(row['site_business_hour'].split('-')[1]) 
                                 else 0, axis=1)
        except:
            print("Impossible de créer Is_business_hour, format de site_business_hour non standard")
    
    # 4. Features d'interaction
    if 'Is_weekend' in df.columns and 'Hour' in df.columns:
        df['Weekend_hour'] = df['Is_weekend'] * df['Hour']
    
    if 'site_criticallity' in df.columns and 'Hour' in df.columns:
        # Convertir site_criticallity en numérique pour l'interaction
        criticality_map = {'low': 1, 'medium': 2, 'high': 3, 'very high': 4}
        df['criticality_numeric'] = df['site_criticallity'].map(criticality_map).fillna(0)
        df['Criticality_hour'] = df['criticality_numeric'] * df['Hour']
    
    # 5. Encodage des variables catégorielles
    categorical_cols = ['ci_type', 'device_role', 'site_criticallity', 'site_country', 'site_city']
    for col in categorical_cols:
        if col in df.columns:
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
    
    # Visualiser la distribution des nouvelles features numériques
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numeric_features[:16]):  # Limiter à 16 features pour la lisibilité
        plt.subplot(4, 4, i+1)
        sns.histplot(df[feature], kde=True)
        plt.title(feature)
    plt.tight_layout()
    plt.savefig(f"TimeSeries_forecasting/cluster_{cluster_id}_feature_distributions.png")
    plt.close()
    
    return df

# Fonction pour la sélection de features
def select_features(X_train, y_train, X_test, cluster_id):
    print(f"Sélection des features pour le cluster {cluster_id}...")
    
    # 1. Corrélation avec la cible
    correlation_scores = {}
    for col in X_train.columns:
        if X_train[col].dtype in ['int64', 'float64']:
            correlation_scores[col] = abs(np.corrcoef(X_train[col], y_train)[0, 1])
    
    correlation_df = pd.DataFrame({
        'Feature': list(correlation_scores.keys()),
        'Correlation': list(correlation_scores.values())
    }).sort_values('Correlation', ascending=False)
    
    # Sauvegarder les scores de corrélation
    correlation_df.to_csv(f"TimeSeries_forecasting/cluster_{cluster_id}_correlation_scores.csv")
    
    # Visualiser les top corrélations
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Correlation', y='Feature', data=correlation_df.head(20))
    plt.title(f'Top 20 Features par Corrélation - Cluster {cluster_id}')
    plt.tight_layout()
    plt.savefig(f"TimeSeries_forecasting/cluster_{cluster_id}_correlation_features.png")
    plt.close()
    
        # 2. Matrice de corrélation pour détecter la multicolinéarité
    corr_matrix = X_train.corr()
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title(f'Matrice de Corrélation - Cluster {cluster_id}')
    plt.tight_layout()
    plt.savefig(f"TimeSeries_forecasting/cluster_{cluster_id}_correlation_matrix.png")
    plt.close()
    
    # Identifier les paires de features hautement corrélées (|corr| > 0.95)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature1', 'Feature2', 'Correlation'])
    high_corr_df.to_csv(f"TimeSeries_forecasting/cluster_{cluster_id}_high_correlation_pairs.csv")
    
    # 3. Feature importance avec Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    feature_importance.to_csv(f"TimeSeries_forecasting/cluster_{cluster_id}_feature_importance.csv")
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title(f'Top 20 Features par Importance - Cluster {cluster_id}')
    plt.tight_layout()
    plt.savefig(f"TimeSeries_forecasting/cluster_{cluster_id}_feature_importance.png")
    plt.close()
    
    # 4. Information mutuelle (pour capturer les relations non linéaires)
    mi_scores = mutual_info_regression(X_train, y_train)
    mi_df = pd.DataFrame({
        'Feature': X_train.columns,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False)
    
    mi_df.to_csv(f"TimeSeries_forecasting/cluster_{cluster_id}_mutual_info_scores.csv")
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='MI_Score', y='Feature', data=mi_df.head(20))
    plt.title(f'Top 20 Features par Information Mutuelle - Cluster {cluster_id}')
    plt.tight_layout()
    plt.savefig(f"TimeSeries_forecasting/cluster_{cluster_id}_mutual_info.png")
    plt.close()
    
    # 5. Sélection des features finales
    # Combiner les scores de différentes méthodes
    combined_scores = pd.DataFrame({'Feature': X_train.columns})
    combined_scores = combined_scores.merge(correlation_df, on='Feature', how='left')
    combined_scores = combined_scores.merge(feature_importance, on='Feature', how='left')
    combined_scores = combined_scores.merge(mi_df, on='Feature', how='left')
    
    # Normaliser les scores
    for col in ['Correlation', 'Importance', 'MI_Score']:
        if col in combined_scores.columns:
            combined_scores[f'{col}_norm'] = combined_scores[col] / combined_scores[col].max()
    
    # Score combiné
    score_cols = [col for col in combined_scores.columns if col.endswith('_norm')]
    combined_scores['Combined_Score'] = combined_scores[score_cols].mean(axis=1)
    combined_scores = combined_scores.sort_values('Combined_Score', ascending=False)
    
    combined_scores.to_csv(f"TimeSeries_forecasting/cluster_{cluster_id}_combined_feature_scores.csv")
    
    # Sélectionner les top features (par exemple, top 50 ou score > 0.1)
    top_features = combined_scores[combined_scores['Combined_Score'] > 0.1]['Feature'].tolist()
    
    # Si trop peu de features sont sélectionnées, prendre les top 50
    if len(top_features) < 50:
        top_features = combined_scores.head(50)['Feature'].tolist()
    
    # Éliminer les features hautement corrélées
    features_to_drop = set()
    for _, row in high_corr_df.iterrows():
        # Garder celle avec le score combiné le plus élevé
        feature1_score = combined_scores[combined_scores['Feature'] == row['Feature1']]['Combined_Score'].values[0]
        feature2_score = combined_scores[combined_scores['Feature'] == row['Feature2']]['Combined_Score'].values[0]
        
        if feature1_score >= feature2_score:
            features_to_drop.add(row['Feature2'])
        else:
            features_to_drop.add(row['Feature1'])
    
    # Filtrer les features finales
    final_features = [f for f in top_features if f not in features_to_drop]
    
    print(f"Nombre de features sélectionnées pour le cluster {cluster_id}: {len(final_features)}")
    
    # Sauvegarder la liste des features finales
    pd.DataFrame({'Feature': final_features}).to_csv(
        f"TimeSeries_forecasting/cluster_{cluster_id}_final_features.csv", index=False)
    
    # Retourner les datasets filtrés
    X_train_filtered = X_train[final_features]
    X_test_filtered = X_test[final_features]
    
    return X_train_filtered, X_test_filtered, final_features

# Fonction pour l'optimisation des hyperparamètres avec Optuna
def optimize_model(X_train, y_train, model_name, cluster_id):
    print(f"Optimisation du modèle {model_name} pour le cluster {cluster_id}...")
    
    # Définir l'espace de recherche des hyperparamètres selon le modèle
    def objective(trial):
        if model_name == 'LinearRegression':
            model = LinearRegression()
            
        elif model_name == 'Ridge':
            alpha = trial.suggest_float('alpha', 0.01, 10.0, log=True)
            model = Ridge(alpha=alpha, random_state=42)
            
        elif model_name == 'Lasso':
            alpha = trial.suggest_float('alpha', 0.001, 1.0, log=True)
            model = Lasso(alpha=alpha, random_state=42)
            
        elif model_name == 'ElasticNet':
            alpha = trial.suggest_float('alpha', 0.001, 1.0, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            
        elif model_name == 'RandomForest':
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
        elif model_name == 'GradientBoosting':
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                subsample=subsample,
                random_state=42
            )
            
        elif model_name == 'XGBoost':
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            model = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=42
            )
            
        elif model_name == 'LightGBM':
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            num_leaves = trial.suggest_int('num_leaves', 20, 100)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            model = LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                num_leaves=num_leaves,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=42
            )
            
        elif model_name == 'CatBoost':
            iterations = trial.suggest_int('iterations', 50, 300)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            depth = trial.suggest_int('depth', 4, 10)
            l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True)
            model = CatBoostRegressor(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                l2_leaf_reg=l2_leaf_reg,
                random_seed=42,
                verbose=0
            )
            
        elif model_name == 'SVR':
            C = trial.suggest_float('C', 0.1, 100.0, log=True)
            epsilon = trial.suggest_float('epsilon', 0.01, 1.0, log=True)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            model = SVR(C=C, epsilon=epsilon, gamma=gamma)
            
        else:
            raise ValueError(f"Modèle {model_name} non supporté")
        
        # Validation croisée temporelle
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            y_pred = model.predict(X_val_cv)
            rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
            cv_scores.append(rmse)
        
        return np.mean(cv_scores)
    
    # Créer l'étude Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    # Récupérer les meilleurs hyperparamètres
    best_params = study.best_params
    best_value = study.best_value
    
    # Sauvegarder les résultats de l'optimisation
    results = {
        'model': model_name,
        'best_params': best_params,
        'best_rmse': best_value
    }
    
    # Visualiser l'importance des hyperparamètres
    if len(best_params) > 1:  # Ne pas visualiser si un seul hyperparamètre
        try:
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(f"TimeSeries_forecasting/cluster_{cluster_id}_{model_name}_param_importance.png")
        except:
            print(f"Impossible de générer le graphique d'importance des hyperparamètres pour {model_name}")
    
    # Créer et entraîner le modèle avec les meilleurs hyperparamètres
    best_model = None
    
    if model_name == 'LinearRegression':
        best_model = LinearRegression()
        
    elif model_name == 'Ridge':
        best_model = Ridge(alpha=best_params['alpha'], random_state=42)
        
    elif model_name == 'Lasso':
        best_model = Lasso(alpha=best_params['alpha'], random_state=42)
        
    elif model_name == 'ElasticNet':
        best_model = ElasticNet(
            alpha=best_params['alpha'], 
            l1_ratio=best_params['l1_ratio'], 
            random_state=42
        )
        
    elif model_name == 'RandomForest':
        best_model = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=42
        )
        
    elif model_name == 'GradientBoosting':
        best_model = GradientBoostingRegressor(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            subsample=best_params['subsample'],
            random_state=42
        )
        
    elif model_name == 'XGBoost':
        best_model = XGBRegressor(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            random_state=42
        )
        
    elif model_name == 'LightGBM':
        best_model = LGBMRegressor(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            num_leaves=best_params['num_leaves'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            random_state=42
        )
        
    elif model_name == 'CatBoost':
        best_model = CatBoostRegressor(
            iterations=best_params['iterations'],
            learning_rate=best_params['learning_rate'],
            depth=best_params['depth'],
            l2_leaf_reg=best_params['l2_leaf_reg'],
            random_seed=42,
            verbose=0
        )
        
    elif model_name == 'SVR':
        best_model = SVR(
            C=best_params['C'],
            epsilon=best_params['epsilon'],
            gamma=best_params['gamma']
        )
    
    # Entraîner le modèle sur l'ensemble des données d'entraînement
    best_model.fit(X_train, y_train)
    
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
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # en pourcentage
    
    # Sauvegarder les métriques
    metrics = {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
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
    test_data = feature_engineering(test_data, cluster_id)
    
    # Si features_df est disponible, fusionner avec les données
    if features_df is not None:
        print("Fusion avec les features d'extraction...")
        train_data = pd.merge(train_data, features_df, on='ci_name', how='left')
        test_data = pd.merge(test_data, features_df, on='ci_name', how='left')
    
    # Préparation des features et de la cible
    target_col = 'event_value'
    
    # Colonnes à exclure des features
    exclude_cols = ['event_value', 'event_date_time', 'Date', 'cluster', 'ci_name', 
                    'Part_of_day', 'site_name', 'site_city', 'site_country', 
                    'site_business_hour', 'service_line']
    
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]
    
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    # 4. Sélection des features
    X_train, X_test, selected_features = select_features(X_train, y_train, X_test, cluster_id)
    
    # 5. Optimisation et évaluation des modèles
    models_to_try = [
        'LinearRegression', 
        'Ridge', 
        'Lasso', 
        'ElasticNet',
        'RandomForest', 
        'GradientBoosting', 
        'XGBoost', 
        'LightGBM'
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
    
    return best_model_info

# Fonction principale
def main():
    # Charger les données
    df, cluster_df, features_df = load_data()
    
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
    
    print("\nTraitement terminé. Tous les résultats ont été sauvegardés dans le dossier 'TimeSeries_forecasting'.")

# Exécution du programme principal
if __name__ == "__main__":
    main()
