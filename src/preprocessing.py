import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess_data(self, df, target_column=None):
        """
        Prétraite les données en suivant les étapes recommandées par l'analyse exploratoire
        """
        df_copy = df.copy()
        
        # Séparation des variables numériques et catégorielles
        numerical_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df_copy.select_dtypes(include=['object']).columns
        
        # Traitement des valeurs manquantes
        if len(numerical_cols) > 0:
            df_copy[numerical_cols] = self.numerical_imputer.fit_transform(df_copy[numerical_cols])
        
        if len(categorical_cols) > 0:
            df_copy[categorical_cols] = self.categorical_imputer.fit_transform(df_copy[categorical_cols])
        
        # Encodage des variables catégorielles
        for col in categorical_cols:
            if col != target_column:
                self.label_encoders[col] = LabelEncoder()
                df_copy[col] = self.label_encoders[col].fit_transform(df_copy[col])
        
        # Normalisation des variables numériques
        if len(numerical_cols) > 0:
            df_copy[numerical_cols] = self.scaler.fit_transform(df_copy[numerical_cols])
        
        return df_copy
    
    def transform_new_data(self, df):
        """
        Applique la même transformation aux nouvelles données
        """
        df_copy = df.copy()
        
        # Séparation des variables numériques et catégorielles
        numerical_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df_copy.select_dtypes(include=['object']).columns
        
        # Application des transformations
        if len(numerical_cols) > 0:
            df_copy[numerical_cols] = self.numerical_imputer.transform(df_copy[numerical_cols])
            df_copy[numerical_cols] = self.scaler.transform(df_copy[numerical_cols])
        
        if len(categorical_cols) > 0:
            df_copy[categorical_cols] = self.categorical_imputer.transform(df_copy[categorical_cols])
            for col in categorical_cols:
                if col in self.label_encoders:
                    df_copy[col] = self.label_encoders[col].transform(df_copy[col])
        
        return df_copy
