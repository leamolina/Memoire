import pandas as pd
import glob
import os

# Dossier des données et modèles de fichiers
data_dir = 'Data_night'
csv_pattern = os.path.join(data_dir, 'data_*.csv')

# Charger la liste des équipements à risque
at_risk_df = pd.read_csv('Data/at_risk_equipment.csv')

at_risk_list = set(at_risk_df['ci_name'])

# Découvrir tous les fichiers Data/data_*.csv (décembre 2024 à juin 2025)
file_paths = sorted(glob.glob(csv_pattern))

monthly_dfs = []
months = []

for fp in file_paths:
    print(f"Traitement du fichier : {fp}")
    fname = os.path.basename(fp)
    month_name = fname.replace('data_', '').replace('.csv', '')
    months.append(month_name)

    df = pd.read_csv(fp)

    # Renommer la colonne si besoin
    if 'max_event_value' in df.columns:
        df = df.rename(columns={'max_event_value': 'event_value'})

    df['event_month'] = month_name.capitalize()

    # Filtrer sur les équipements à risque
    df = df[df['ci_name'].isin(at_risk_list)].copy()
    monthly_dfs.append(df)

# Concaténation et filtrage des CI présents tous les mois
print("Vérification de la présence des équipements à risque sur tous les mois...")
df_concat = pd.concat(monthly_dfs, ignore_index=True)

counts_per_ci = (
    df_concat.groupby('ci_name')['event_month']
    .nunique()
    .reset_index(name='months_present')
)
n_months = len(set(months))
cis_all_months = set(
    counts_per_ci.loc[counts_per_ci['months_present'] == n_months, 'ci_name']
)
data_complete = df_concat[df_concat['ci_name'].isin(cis_all_months)].copy()

# Mapping mois → année
year_map = {
    'December': 2024,
    'January': 2025,
    'February': 2025,
    'March': 2025,
    'April': 2025,
    'May': 2025,
    'June': 2025
}

# 1. S’assurer que event_day et event_hour sont bien des entiers
data_complete['event_day'] = data_complete['event_day'].astype(int)
data_complete['event_hour'] = data_complete['event_hour'].astype(int)

# 2. Construire event_date (date sans heure)
data_complete['event_date'] = data_complete.apply(
    lambda r: pd.to_datetime(
        f"{r['event_day']} {r['event_month']} {year_map[r['event_month']]}",
        dayfirst=True
    ),
    axis=1
)

# 3. Ajouter event_date_time en ajoutant l’heure
data_complete['event_date_time'] = (
    data_complete['event_date']
    + pd.to_timedelta(data_complete['event_hour'], unit='h')
)

# 4. (Optionnel) Re-vérification et nettoyage
print(data_complete[['event_day', 'event_hour', 'event_date', 'event_date_time']].head())
print("Type de event_date_time :", data_complete['event_date_time'].dtype)
print("Dates-temps manquantes :", data_complete['event_date_time'].isna().sum())

# 5. Sauvegarde
print("Sauvegarde des fichiers...")
dir_out = data_dir
data_complete.to_csv(os.path.join(dir_out, 'data_complete.csv'), index=False)

data_complete_interface = data_complete[data_complete['ci_type'] == 'Interface']
data_complete_interface.to_csv(
    os.path.join(dir_out, 'data_complete_interface.csv'), index=False
)

data_complete_logical = data_complete[data_complete['ci_type'] == 'Logical Connection']
data_complete_logical.to_csv(
    os.path.join(dir_out, 'data_complete_logical_connection.csv'), index=False
)

print("Fichiers générés :")
print(" - data_complete.csv (tous CI présents tous mois, avec date et heure)")
print(" - data_complete_interface.csv (Interface)")
print(" - data_complete_logical_connection.csv (Logical Connection)")
