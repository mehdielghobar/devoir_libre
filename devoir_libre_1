import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

#  PARTIE 1: TROUVER LES VARIABLES MANQUANTES

print("="*60)
print("PARTIE 1: ANALYSE DES DONNÉES MANQUANTES")
print("="*60)

# 1.1 - Charger et visualiser les données
print("\n1.1 - Chargement du dataset Titanic:")
titanic = pd.read_csv('titanic_survival.csv')
print("\nPremières 5 lignes du dataset:")
print(titanic.head())

# 1.2 - Taille du dataset et variables
print("\n1.2 - Informations sur le dataset:")
print(f"Taille du dataset: {titanic.shape[0]} lignes × {titanic.shape[1]} colonnes")
print(f"\nColonnes (caractéristiques): {titanic.columns.tolist()}")
print(f"\nVariable cible: 'survived'")
print(f"\nY a-t-il des données manquantes? {titanic.isnull().any().any()}")

# 1.3 - Compter les valeurs manquantes dans 'age'
print("\n1.3 - Valeurs manquantes dans la colonne 'age':")
age = titanic['age']
missing_values = age[pd.isnull(age)]
missing_values_count = len(missing_values)
print(f"Nombre de valeurs manquantes: {missing_values_count}")

# 1.4 - Compter les valeurs manquantes dans 'cabin'
print("\n1.4 - Valeurs manquantes dans la colonne 'cabin':")
print(f"Nombre de valeurs manquantes: {titanic['cabin'].isnull().sum()}")

# 1.5 - Compter pour toutes les colonnes
print("\n1.5 - Valeurs manquantes par colonne:")
missing_per_column = titanic.isnull().sum()
print(missing_per_column[missing_per_column > 0])

# 1.6 - Discussion
print("\n1.6 - IMPORTANCE DE GÉRER LES DONNÉES MANQUANTES:")
print("""
✓ Les modèles ML ne peuvent pas traiter directement les valeurs manquantes
✓ Les données manquantes peuvent biaiser les résultats
✓ Méthodes de gestion:
   • Suppression: lignes ou colonnes avec trop de valeurs manquantes
   • Imputation numérique: moyenne, médiane (age)
   • Imputation catégorique: mode, 'Inconnu' (embarked, cabin)
   • Méthodes avancées: KNN, régression
""")

#  PARTIE 2: GÉRER LES VARIABLES MANQUANTES

print("\n" + "="*60)
print("PARTIE 2: TRAITEMENT DES VALEURS MANQUANTES")
print("="*60)

titanic_clean = titanic.copy()

# 2.1 - Supprimer les lignes avec valeurs manquantes dans 'embarked'
print("\n2.1 - Suppression des lignes avec 'embarked' manquant:")
avant = titanic_clean.shape[0]
titanic_clean = titanic_clean.dropna(subset=['embarked'])
apres = titanic_clean.shape[0]
print(f"Lignes supprimées: {avant - apres}")

# 2.2 - Supprimer la colonne 'cabin'
print("\n2.2 - Suppression de la colonne 'cabin':")
titanic_clean = titanic_clean.drop(columns=['cabin'])
print(f"Colonnes restantes: {len(titanic_clean.columns)}")

# 2.3 - Imputation
print("\n2.3 - Imputation des valeurs manquantes:")
# Imputation numérique - Age par la moyenne
age_mean = titanic_clean['age'].mean()
titanic_clean['age'] = titanic_clean['age'].fillna(age_mean)
print(f"  • Age: remplacé par la moyenne ({age_mean:.2f})")

# Imputation catégorique - Embarked par le mode
embarked_mode = titanic_clean['embarked'].mode()[0]
titanic_clean['embarked'] = titanic_clean['embarked'].fillna(embarked_mode)
print(f"  • Embarked: remplacé par le mode ({embarked_mode})")

#  PARTIE 3: GÉRER LES VARIABLES CATÉGORIQUES

print("\n" + "="*60)
print("PARTIE 3: ENCODAGE DES VARIABLES CATÉGORIQUES")
print("="*60)

# 3.1 - Encodage de 'embarked' (variable indépendante)
print("\n3.1 - Encodage de 'embarked' (OneHotEncoder):")
print("Pourquoi OneHotEncoder? Car 'embarked' n'a pas d'ordre naturel (S, C, Q)")
onehot = OneHotEncoder(sparse_output=False, drop='first')
embarked_encoded = onehot.fit_transform(titanic_clean[['embarked']])
embarked_df = pd.DataFrame(embarked_encoded, 
                           columns=onehot.get_feature_names_out(['embarked']),
                           index=titanic_clean.index)
titanic_clean = pd.concat([titanic_clean, embarked_df], axis=1)
titanic_clean = titanic_clean.drop(columns=['embarked'])
print(f"Nouvelles colonnes créées: {list(embarked_df.columns)}")

# 3.2 - Encodage de 'sex' (variable dépendante)
print("\n3.2 - Encodage de 'sex' (LabelEncoder):")
print("Pourquoi LabelEncoder? Car c'est une variable binaire")
le = LabelEncoder()
titanic_clean['sex_encoded'] = le.fit_transform(titanic_clean['sex'])
print(f"Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

#  PARTIE 4: SÉLECTION DES CARACTÉRISTIQUES 

print("\n" + "="*60)
print("PARTIE 4: SÉLECTION DES CARACTÉRISTIQUES")
print("="*60)

# 4.1 - Sélection des colonnes
print("\n4.1 - Sélection des colonnes pertinentes:")
selected_columns = ['pclass', 'sex_encoded', 'age', 'fare', 'survived']
titanic_final = titanic_clean[selected_columns].copy()
print(f"Colonnes sélectionnées: {selected_columns}")

# 4.2 - Méthodes d'identification des caractéristiques importantes
print("\n4.2 - Comment identifier les caractéristiques importantes?")
print("""
Méthodes statistiques:
   • Corrélation avec la variable cible
   • Test du chi-2 pour variables catégoriques
   
Méthodes basées sur les modèles:
   • Feature importance (Random Forest, XGBoost)
   • Coefficients de régression logistique
   • LASSO (L1) pour sélection automatique
   
Méthodes de filtrage:
   • Mutual information
   • Variance threshold
   
Connaissances métier:
   • Importance contextuelle des variables
   • Analyse exploratoire des données
""")

# 4.3 - Vérification finale
print("\n4.3 - Vérification des données:")
print(f"Valeurs manquantes: {titanic_final.isnull().sum().sum()}")
print(f"Types de données:\n{titanic_final.dtypes}")

#  PARTIE 5: DIVISION DES DONNÉES 

print("\n" + "="*60)
print("PARTIE 5: DIVISION TRAIN/TEST")
print("="*60)

# 5.1 & 5.2 - Division et affichage
print("\n5.1 & 5.2 - Division des données:")
X = titanic_final.drop('survived', axis=1)
y = titanic_final['survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# 5.3 - Importance de la division
print("\n5.3 - Pourquoi diviser les données?")
print("""
✓ Évite le surapprentissage (overfitting)
✓ Permet d'évaluer la généralisation du modèle
✓ Donne une estimation réaliste des performances
✓ Permet d'optimiser les hyperparamètres sans biais
""")

#  PARTIE 6: FEATURE SCALING 

print("\n" + "="*60)
print("PARTIE 6: NORMALISATION DES CARACTÉRISTIQUES")
print("="*60)

# 6.1 - Normalisation
print("\n6.1 - Application des scalers:")

# StandardScaler
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

print("\n--- STANDARD SCALER (Standardisation) ---")
print(f"Moyennes après scaling: {X_train_std.mean(axis=0)}")
print(f"Écarts-types après scaling: {X_train_std.std(axis=0)}")
print(f"Plage des valeurs: [{X_train_std.min():.2f}, {X_train_std.max():.2f}]")

# MinMaxScaler
scaler_mm = MinMaxScaler()
X_train_mm = scaler_mm.fit_transform(X_train)
X_test_mm = scaler_mm.transform(X_test)

print("\n--- MIN MAX SCALER (Normalisation) ---")
print(f"Moyennes après scaling: {X_train_mm.mean(axis=0)}")
print(f"Écarts-types après scaling: {X_train_mm.std(axis=0)}")
print(f"Plage des valeurs: [{X_train_mm.min():.2f}, {X_train_mm.max():.2f}]")

print("\n" + "="*60)
print("DIFFÉRENCES ENTRE STANDARDSCALER ET MINMAXSCALER:")
print("="*60)
print("""
STANDARDSCALER:
   • Centre les données (moyenne = 0, écart-type = 1)
   • Plage: environ [-3, 3] (selon les outliers)
   • Préserve la forme de la distribution
   • Recommandé pour: données gaussiennes, PCA, SVM, régression linéaire

MINMAXSCALER:
   • Met les données dans une plage fixe [0, 1]
   • Préserve les relations mais change l'échelle
   • Recommandé pour: réseaux de neurones, KNN, K-means
   • Plus sensible aux outliers
""")

print("\n DEVOIR COMPLÉTÉ AVEC SUCCÈS!")
