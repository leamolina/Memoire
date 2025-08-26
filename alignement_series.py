

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta


def analyze_date_distribution(df):
    """Analyse la distribution des dates par ci_name.
    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données
    Returns:
        pandas.DataFrame: Un DataFrame avec les statistiques par ci_name
    """
    # S'assurer que event_date_time est au format datetime
    if not pd.api.types.is_datetime64_any_dtype(df['event_date_time']):
        df['event_date_time'] = pd.to_datetime(df['event_date_time'])

    # Nombre de dates uniques par ci_name
    date_counts = df.groupby('ci_name')['event_date_time'].nunique()

    # Statistiques globales
    print("Statistiques globales sur le nombre de dates par appareil:")
    print(f"Moyenne: {date_counts.mean():.2f}")
    print(f"Médiane: {date_counts.median():.2f}")
    print(f"Écart-type: {date_counts.std():.2f}")
    print(f"Minimum: {date_counts.min()}")
    print(f"Maximum: {date_counts.max()}")
    print(f"25ème percentile: {date_counts.quantile(0.25)}")
    print(f"75ème percentile: {date_counts.quantile(0.75)}")

    # Visualiser la distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(date_counts, bins=50, kde=True)
    plt.title("Distribution du nombre de dates par appareil")
    plt.xlabel("Nombre de dates uniques")
    plt.ylabel("Nombre d'appareils")
    plt.axvline(date_counts.mean(), color='r', linestyle='--', label=f'Moyenne: {date_counts.mean():.2f}')
    plt.axvline(date_counts.median(), color='g', linestyle='--', label=f'Médiane: {date_counts.median():.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Analyse_alignement_series_night/date_distribution.png', dpi=300)
    plt.close()

    # Créer un DataFrame pour stocker les statistiques détaillées par ci_name
    stats_df = pd.DataFrame()

    # Calculer les statistiques pour chaque ci_name
    for name, group in df.groupby('ci_name'):
        dates = group['event_date_time'].sort_values().unique()
        min_date = dates.min()
        max_date = dates.max()
        total_period = (max_date - min_date).days + 1
        unique_dates = len(dates)
        coverage = unique_dates / total_period * 100 if total_period > 0 else 0

        if len(dates) > 1:
            date_diffs = [(dates[i+1] - dates[i]).total_seconds() / (60*60*24)
                          for i in range(len(dates)-1)]
            avg_gap = np.mean(date_diffs)
            max_gap = np.max(date_diffs)
        else:
            avg_gap = np.nan
            max_gap = np.nan

        unique_weeks = len(pd.DatetimeIndex(dates).isocalendar().week.unique())
        unique_months = len(pd.DatetimeIndex(dates).month.unique())

        stats_df.loc[name, 'unique_dates'] = unique_dates
        stats_df.loc[name, 'min_date'] = min_date
        stats_df.loc[name, 'max_date'] = max_date
        stats_df.loc[name, 'period_days'] = total_period
        stats_df.loc[name, 'coverage_percent'] = coverage
        stats_df.loc[name, 'avg_gap_days'] = avg_gap
        stats_df.loc[name, 'max_gap_days'] = max_gap
        stats_df.loc[name, 'unique_weeks'] = unique_weeks
        stats_df.loc[name, 'unique_months'] = unique_months

    # Calculer la période globale
    global_min_date = df['event_date_time'].min()
    global_max_date = df['event_date_time'].max()
    global_period_days = (global_max_date - global_min_date).days + 1
    global_weeks = pd.DatetimeIndex(df['event_date_time']).isocalendar().week.unique()
    global_months = pd.DatetimeIndex(df['event_date_time']).month.unique()

    print(f"\nPériode globale de l'étude: du {global_min_date.date()} au {global_max_date.date()} ({global_period_days} jours)")
    print(f"Nombre total de semaines: {len(global_weeks)}")
    print(f"Nombre total de mois: {len(global_months)}")

    # Statistiques sur la couverture
    print("\nStatistiques sur la couverture temporelle des appareils:")
    print(f"Couverture moyenne: {stats_df['coverage_percent'].mean():.2f}%")
    print(f"Écart moyen entre dates consécutives: {stats_df['avg_gap_days'].mean():.2f} jours")
    print(f"Écart maximal moyen: {stats_df['max_gap_days'].mean():.2f} jours")

    # Identifier les appareils avec couverture complète ou faible
    high_coverage = stats_df[stats_df['coverage_percent'] >= 90]
    print(f"\nNombre d'appareils avec une couverture ≥90%: {len(high_coverage)} ({len(high_coverage)/len(stats_df)*100:.2f}%)")
    low_coverage = stats_df[stats_df['coverage_percent'] <= 10]
    print(f"Nombre d'appareils avec une couverture ≤10%: {len(low_coverage)} ({len(low_coverage)/len(stats_df)*100:.2f}%)")

    # Visualisations supplémentaires
    plt.figure(figsize=(12, 6))
    sns.histplot(stats_df['coverage_percent'], bins=50, kde=True)
    plt.title("Distribution du pourcentage de couverture temporelle par appareil")
    plt.xlabel("Pourcentage de couverture")
    plt.ylabel("Nombre d'appareils")
    plt.axvline(stats_df['coverage_percent'].mean(), color='r', linestyle='--',
                label=f'Moyenne: {stats_df["coverage_percent"].mean():.2f}%')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Analyse_alignement_series_night/coverage_distribution.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.histplot(stats_df['avg_gap_days'].dropna(), bins=50, kde=True)
    plt.title("Distribution de l'écart moyen entre dates consécutives par appareil")
    plt.xlabel("Écart moyen (jours)")
    plt.ylabel("Nombre d'appareils")
    plt.axvline(stats_df['avg_gap_days'].mean(), color='r', linestyle='--',
                label=f'Moyenne: {stats_df["avg_gap_days"].mean():.2f} jours')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Analyse_alignement_series_night/avg_gap_distribution.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='unique_dates', y='coverage_percent', data=stats_df)
    plt.title("Relation entre le nombre de dates et la couverture temporelle")
    plt.xlabel("Nombre de dates uniques")
    plt.ylabel("Pourcentage de couverture")
    plt.tight_layout()
    plt.savefig('Analyse_alignement_series_night/dates_vs_coverage.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.countplot(x='unique_weeks', data=stats_df)
    plt.title("Distribution du nombre de semaines uniques par appareil")
    plt.xlabel("Nombre de semaines uniques")
    plt.ylabel("Nombre d'appareils")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('Analyse_alignement_series_night/unique_weeks_distribution.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.countplot(x='unique_months', data=stats_df)
    plt.title("Distribution du nombre de mois uniques par appareil")
    plt.xlabel("Nombre de mois uniques")
    plt.ylabel("Nombre d'appareils")
    plt.tight_layout()
    plt.savefig('Analyse_alignement_series_night/unique_months_distribution.png', dpi=300)
    plt.close()

    return stats_df


def analyze_weekly_presence(df):
    # S'assurer que event_date_time est au format datetime
    df['event_date_time'] = pd.to_datetime(df['event_date_time'])

    # Ajouter la colonne semaine et année
    df['week'] = df['event_date_time'].dt.isocalendar().week
    df['year'] = df['event_date_time'].dt.isocalendar().year

    # Date représentative pour chaque semaine (premier jour)
    df['week_start'] = df.apply(
        lambda x: pd.to_datetime(f"{x['year']}-W{x['week']:02d}-1", format='%Y-W%W-%w'),
        axis=1
    )

    total_devices = df['ci_name'].nunique()
    print(f"Nombre total d'appareils uniques: {total_devices}")

    # Compter le nombre d'appareils uniques par semaine
    weekly_devices = df.groupby('week_start')['ci_name'].nunique()
    weekly_percentage = (weekly_devices / total_devices * 100).round(2)

    weekly_stats = pd.DataFrame({
        'week_start': weekly_devices.index,
        'devices_count': weekly_devices.values,
        'percentage': weekly_percentage.values
    }).sort_values('week_start')

    weekly_stats['year_week_label'] = weekly_stats['week_start'].dt.strftime('%Y-%W')

    print("\nPourcentage d'appareils présents par semaine:")
    for _, row in weekly_stats.iterrows():
        print(f"Semaine {row['year_week_label']}: {row['devices_count']} appareils ({row['percentage']}%)")

    print("\nStatistiques sur la présence hebdomadaire:")
    print(f"Moyenne: {weekly_percentage.mean():.2f}%")
    print(f"Médiane: {weekly_percentage.median():.2f}%")
    print(f"Minimum: {weekly_percentage.min():.2f}%")
    print(f"Maximum: {weekly_percentage.max():.2f}%")

    plt.figure(figsize=(15, 6))
    order = weekly_stats['year_week_label'].tolist()
    sns.barplot(x='year_week_label', y='percentage', data=weekly_stats, order=order)
    plt.title("Pourcentage d'appareils présents par semaine")
    plt.xlabel("Semaine")
    plt.ylabel("Pourcentage d'appareils")
    plt.axhline(y=weekly_percentage.mean(), color='r', linestyle='--',
                label=f'Moyenne: {weekly_percentage.mean():.2f}%')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Analyse_alignement_series_night/weekly_presence_percentage.png', dpi=300)
    plt.close()

    # Analyser la constance de présence
    df['year_week'] = df['event_date_time'].dt.strftime('%Y-%W')
    device_weeks = df.groupby('ci_name')['year_week'].nunique()
    total_weeks = df['year_week'].nunique()
    device_presence = (device_weeks / total_weeks * 100).round(2)

    print(f"\nNombre total de semaines dans les données: {total_weeks}")
    print("\nStatistiques sur la présence des appareils:")
    print(f"Pourcentage moyen de semaines où un appareil est présent: {device_presence.mean():.2f}%")
    print(f"Médiane: {device_presence.median():.2f}%")
    print(f"Minimum: {device_presence.min():.2f}%")
    print(f"Maximum: {device_presence.max():.2f}%")

    presence_categories = pd.cut(
        device_presence,
        bins=[0, 10, 25, 50, 75, 90, 100],
        labels=['0-10%', '10-25%', '25-50%', '50-75%', '75-90%', '90-100%']
    )
    category_counts = presence_categories.value_counts().sort_index()

    print("\nDistribution des appareils par catégorie de présence:")
    for category, count in category_counts.items():
        print(f"{category}: {count} appareils ({count/total_devices*100:.2f}%)")

    plt.figure(figsize=(12, 6))
    sns.countplot(x=presence_categories, order=category_counts.index)
    plt.title("Distribution des appareils par catégorie de présence hebdomadaire")
    plt.xlabel("Pourcentage de semaines où l'appareil est présent")
    plt.ylabel("Nombre d'appareils")
    plt.tight_layout()
    plt.savefig('Analyse_alignement_series_night/device_presence_categories.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.histplot(device_presence, bins=50, kde=True)
    plt.title("Distribution du pourcentage de semaines où chaque appareil est présent")
    plt.xlabel("Pourcentage de semaines")
    plt.ylabel("Nombre d'appareils")
    plt.axvline(device_presence.mean(), color='r', linestyle='--',
                label=f'Moyenne: {device_presence.mean():.2f}%')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Analyse_alignement_series_night/device_presence_distribution.png', dpi=300)
    plt.close()

    presence_matrix = pd.crosstab(df['ci_name'], df['year_week'])
    presence_matrix = presence_matrix.applymap(lambda x: 1 if x > 0 else 0)
    week_correlation = presence_matrix.corr()

    plt.figure(figsize=(15, 12))
    sns.heatmap(week_correlation, cmap="YlGnBu", vmin=-1, vmax=1)
    plt.title("Corrélation de la présence des appareils entre les semaines")
    plt.tight_layout()
    plt.savefig('Analyse_alignement_series_night/week_correlation_matrix.png', dpi=300)
    plt.close()

    device_presence_df = pd.DataFrame({
        'ci_name': device_weeks.index,
        'weeks_present': device_weeks.values,
        'total_weeks': total_weeks,
        'presence_percentage': device_presence.values,
        'presence_category': presence_categories.values
    })

    return weekly_stats, device_presence_df


def plots_supplementaires(df):
    # On part de data_complete avec au moins les colonnes
    # ['event_date_time', 'ci_name', 'event_value']

    # 1. Pivot : ligne = timestamp, colonnes = ci_name, valeurs = event_value (ou autre champ)
    pivot = df.pivot(
        index='event_date_time',
        columns='ci_name',
        values='event_value'
    )

    # 2. Comptage des valeurs manquantes par équipement
    missing_per_ci = pivot.isna().sum()
    print("Nombre de timestamps manquants par équipement :")
    print(missing_per_ci[missing_per_ci > 0])

    # 3. Comptage des timestamps auxquel·les il manque au moins un équipement
    missing_per_timestamp = pivot.isna().any(axis=1).sum()
    print(f"\nNombre de timestamps où au moins une série est absente : {missing_per_timestamp}")

    # 4. Vérification globale : matrice complète ?
    n_timestamps = pivot.shape[0]
    n_cis       = pivot.shape[1]
    n_rows      = len(df)

    print(f"\nRésultat attendu si pas de lacune : {n_timestamps} timestamps × {n_cis} équipements = {n_timestamps * n_cis} lignes")
    print(f"Vous avez           : {n_rows} lignes dans data_complete")

    if missing_per_ci.sum() == 0:
        print("\n✅ Toutes les séries sont bien alignées (aucune valeur manquante).")
    else:
        print("\n⚠️ Des alignements sont manquants. Inspectez les résultats ci‑dessus.")


    # Tracer l’histogramme
    plt.figure()
    plt.hist(missing_per_ci.values, bins=50)
    plt.xlabel("Nombre de timestamps manquants")
    plt.ylabel("Nombre d'équipements (CI)")
    plt.title("Distribution des données manquantes par équipement")
    plt.tight_layout()
    plt.savefig("Analyse_alignement_series_night/distribution_donnesmanquantes_par_ci.png")


    plt.figure(figsize=(12, 8))
    # 1 pour manquant, 0 pour présent
    mat = pivot.isna().astype(int).T.values  
    plt.imshow(mat, aspect='auto', interpolation='none', cmap='Greys')
    plt.xlabel("Horodatages")
    plt.ylabel("CI")
    plt.title("Heatmap de présence (blanc) vs absence (noir)")
    plt.tight_layout()
    plt.savefig("Analyse_alignement_series_night/heatmap_presence_absence.png")


    plt.figure(figsize=(10, 6))
    pct_missing = missing_per_ci / pivot.shape[0] * 100
    pct_missing.sort_values(ascending=False).plot.bar(figsize=(12,6))
    plt.ylabel("% de timestamps manquants")
    plt.title("Taux de missingness par CI")
    plt.tight_layout()
    plt.savefig("Analyse_alignement_series_night/taux_missingness.png")

    plt.figure(figsize=(12, 4))
    missing_count = pivot.isna().sum(axis=1)
    missing_count.plot()
    plt.ylabel("Nombre de CI manquants")
    plt.title("Évolution du nombre de CI absents dans le temps")
    plt.tight_layout()
    plt.savefig("Analyse_alignement_series_night/evolution_ci_manquants.png")

    lengths = []
    for col in pivot:
        arr = pivot[col].isna().values
        i = 0
        while i < len(arr):
            if arr[i]:
                j = i
                while j < len(arr) and arr[j]:
                    j += 1
                lengths.append(j - i)
                i = j
            else:
                i += 1

    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=50)
    plt.xlabel("Durée consécutive d'absences (en timestamps)")
    plt.ylabel("Fréquence")
    plt.title("Distribution des longueurs de trous par CI")
    plt.tight_layout()
    plt.savefig("Analyse_alignement_series_night/distribution_longueur_trous.png")



# Chargement et analyse des données
df = pd.read_csv('Data_night/data_complete_interface.csv')

if not pd.api.types.is_datetime64_any_dtype(df['event_date_time']):
    print("Conversion de event_date_time en format datetime...")
    df['event_date_time'] = pd.to_datetime(df['event_date_time'])

print("Aperçu des données:")
print(df.head())
print("\nInformations sur les colonnes:")
print(df.info())
print("\nStatistiques descriptives:")
print(df.describe())

# Ajout de colonnes de criticité
df['Criticality_unknown'] = (df['site_criticality'] == 'Unknown').astype(int)
print("value count ", df['Criticality_unknown'].value_counts())

# Vérifier les valeurs manquantes
print("\nValeurs manquantes par colonne:")
print(df.isnull().sum())

stats_df = analyze_date_distribution(df)
stats_df.to_csv('Analyse_alignement_series_night/date_distribution_stats.csv')

weekly_stats, device_presence = analyze_weekly_presence(df)

print('nombre d equipements présents sur la majorité des semaines:',
      len(device_presence[device_presence['presence_percentage'] >= 50]))

print("\nTop 10 des appareils avec la présence la plus constante:")
print(device_presence.sort_values('presence_percentage', ascending=False).head(10))
