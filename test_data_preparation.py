import pandas as pd
import numpy as np

# Charger les données
general_data = pd.read_csv("general_data.csv")
manager_survey_data = pd.read_csv("manager_survey_data.csv")
employee_survey_data = pd.read_csv("employee_survey_data.csv")
in_time = pd.read_csv("in_time.csv")
out_time = pd.read_csv("out_time.csv")

# Aperçu des premières lignes des fichiers
general_data.head()
manager_survey_data.head()
employee_survey_data.head()
in_time.head()
out_time.head()

# Étape 1 : Nettoyage de general_data
print("Valeurs manquantes par colonne :")
print(general_data.isnull().sum())

num_cols = general_data.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if general_data[col].isnull().sum() > 0:
        general_data[col].fillna(general_data[col].mean(), inplace=True)

cat_cols = general_data.select_dtypes(include=["object"]).columns
for col in cat_cols:
    if general_data[col].isnull().sum() > 0:
        general_data[col].fillna(general_data[col].mode()[0], inplace=True)

columns_to_drop = ["EmployeeCount", "Over18", "StandardHours"]
general_data.drop(columns=columns_to_drop, inplace=True)

categorical_cols = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus"]
general_data = pd.get_dummies(general_data, columns=categorical_cols, drop_first=True)

print("Distribution de la cible Attrition :")
print(general_data["Attrition"].value_counts())

general_data["Attrition"] = general_data["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)

corr_matrix = general_data.corr()
print("Matrice de corrélation avec Attrition :")
print(corr_matrix["Attrition"].sort_values(ascending=False))

# Nettoyage de manager_survey_data
print("Valeurs manquantes dans manager_survey_data :")
print(manager_survey_data.isnull().sum())
print("Vérification de l'unicité de EmployeeID dans manager_survey_data :", manager_survey_data["EmployeeID"].is_unique)

merged_data = pd.merge(general_data, manager_survey_data, on="EmployeeID", how="inner")

print("Dimensions après fusion :", merged_data.shape)
print("Valeurs manquantes après fusion :")
print(merged_data[["JobInvolvement", "PerformanceRating"]].isnull().sum())

# Nettoyage de employee_survey_data
print("Valeurs manquantes dans employee_survey_data :")
print(employee_survey_data.isnull().sum())

survey_num_cols = ["EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance"]
for col in survey_num_cols:
    if employee_survey_data[col].isnull().sum() > 0:
        employee_survey_data[col].fillna(employee_survey_data[col].mean(), inplace=True)

merged_data = pd.merge(merged_data, employee_survey_data, on="EmployeeID", how="inner")

print("Dimensions après fusion avec employee_survey_data :", merged_data.shape)
print("Valeurs manquantes après la deuxième fusion :")
print(merged_data.isnull().sum())

# Traitement des fichiers in_time et out_time
in_time.rename(columns={"Unnamed: 0": "EmployeeID"}, inplace=True)
out_time.rename(columns={"Unnamed: 0": "EmployeeID"}, inplace=True)

# Conversion des colonnes de dates en format datetime
in_time.iloc[:, 1:] = in_time.iloc[:, 1:].apply(pd.to_datetime, errors='coerce')
out_time.iloc[:, 1:] = out_time.iloc[:, 1:].apply(pd.to_datetime, errors='coerce')

# Calcul des heures travaillées par jour
work_hours = out_time.iloc[:, 1:] - in_time.iloc[:, 1:]
work_hours = work_hours.applymap(lambda x: x.total_seconds() / 3600 if pd.notnull(x) else np.nan)

# Calcul des métriques
average_hours_per_day = work_hours.mean(axis=1)
irregular_days = work_hours.isnull().sum(axis=1)

# Ajout des métriques au dataframe in_time
work_metrics = pd.DataFrame({
    "EmployeeID": in_time["EmployeeID"],
    "AverageHoursPerDay": average_hours_per_day,
    "IrregularDays": irregular_days
})

# Fusionner avec merged_data
merged_data = pd.merge(merged_data, work_metrics, on="EmployeeID", how="inner")

# Identification et séparation des colonnes combinées
def identify_and_split_combined_columns(df):
    for col in df.select_dtypes(include=["object"]):
        if df[col].str.contains(",").any():
            print(f"Séparation de la colonne combinée : {col}")
            split_columns = df[col].str.split(',', expand=True)
            split_columns.columns = [f"{col}_part{i+1}" for i in range(split_columns.shape[1])]
            df = pd.concat([df.drop(columns=[col]), split_columns], axis=1)
    return df

# Appliquer la fonction après la fusion finale
merged_data = identify_and_split_combined_columns(merged_data)

# Sauvegarde des données finales
merged_data.to_csv("final_merged_data_with_work_metrics.csv", index=False)

print("Fusion finale terminée. Les données complètes avec colonnes séparées sont sauvegardées dans 'final_merged_data_with_work_metrics.csv'.")
