import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.agents.agent_coordinator import AgentCoordinator

# Configuration de la page
st.set_page_config(page_title="Analyse des Données de Risque de Crédit", layout="wide")
st.title("Analyse Exploratoire des Données de Risque de Crédit")

# Initialisation du coordinateur d'agents
@st.cache_resource
def get_agent_coordinator():
    # Spécifiez ici le chemin vers votre modèle GPT4All
    model_path = "models/ggml-gpt4all-j-v1.3-groovy.bin"
    return AgentCoordinator(model_path)

try:
    agent_coordinator = get_agent_coordinator()
    st.sidebar.success("✅ Agents IA initialisés avec succès!")
except Exception as e:
    st.sidebar.error(f"❌ Erreur lors de l'initialisation des agents: {str(e)}")
    st.sidebar.info("👉 Assurez-vous d'avoir téléchargé le modèle GPT4All dans le dossier 'models'")

# Upload du fichier
st.header("Upload des Données")
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    # Chargement des données
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Fichier chargé avec succès!")
        
        # Obtenir les recommandations de l'agent d'analyse
        analysis_recommendations = agent_coordinator.get_analysis_recommendations(df)
        
        # Affichage des informations générales
        st.header("1. Aperçu des Données")
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.subheader("Premières lignes du dataset")
            st.dataframe(df.head())

        with col2:
            st.subheader("Dimensions du dataset")
            st.write(f"Nombre de lignes: {df.shape[0]}")
            st.write(f"Nombre de colonnes: {df.shape[1]}")

        with col3:
            st.subheader("Types de données")
            st.write("Numériques:", len(df.select_dtypes(include=['float64', 'int64']).columns))
            st.write("Catégorielles:", len(df.select_dtypes(include=['object']).columns))

        # Information détaillée sur les colonnes
        st.subheader("Information sur les colonnes")
        info_df = pd.DataFrame({
            'Type de données': df.dtypes,
            'Valeurs manquantes': df.isnull().sum(),
            'Pourcentage manquant': (df.isnull().sum() / len(df) * 100).round(2),
            'Valeurs uniques': df.nunique()
        })
        st.dataframe(info_df)

        # Obtenir les recommandations de prétraitement
        preprocessing_recommendations = agent_coordinator.get_preprocessing_recommendations(
            df, analysis_recommendations
        )

        # Statistiques descriptives
        st.header("2. Statistiques Descriptives")
        st.dataframe(df.describe().round(2))

        # Analyse des variables catégorielles
        st.header("3. Distribution des Variables Catégorielles")
        categorical_cols = df.select_dtypes(include=['object']).columns

        if len(categorical_cols) > 0:
            for col in categorical_cols:
                fig = px.pie(df, names=col, title=f'Distribution de {col}')
                st.plotly_chart(fig)
        else:
            st.info("Aucune variable catégorielle trouvée dans le dataset")

        # Analyse des variables numériques
        st.header("4. Distribution des Variables Numériques")
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

        if len(numerical_cols) > 0:
            cols = st.columns(2)
            for idx, col in enumerate(numerical_cols):
                with cols[idx % 2]:
                    fig = px.histogram(df, x=col, title=f'Distribution de {col}')
                    st.plotly_chart(fig)

                    # Statistiques de base
                    stats = df[col].describe()
                    st.write(f"Moyenne: {stats['mean']:.2f}")
                    st.write(f"Médiane: {stats['50%']:.2f}")
                    st.write(f"Écart-type: {stats['std']:.2f}")
        else:
            st.info("Aucune variable numérique trouvée dans le dataset")

        # Matrice de corrélation
        st.header("5. Analyse des Corrélations")
        numerical_df = df.select_dtypes(include=['float64', 'int64'])
        
        if len(numerical_df.columns) > 1:
            correlation_matrix = numerical_df.corr()

            fig = px.imshow(correlation_matrix,
                          labels=dict(color="Corrélation"),
                          title="Matrice de Corrélation")
            st.plotly_chart(fig)

            # Identification des corrélations fortes
            st.subheader("5.1 Paires de Variables Fortement Corrélées")
            correlation_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if abs(correlation_matrix.iloc[i,j]) > 0.5:  # Seuil de corrélation
                        correlation_pairs.append({
                            'Variable 1': correlation_matrix.columns[i],
                            'Variable 2': correlation_matrix.columns[j],
                            'Corrélation': correlation_matrix.iloc[i,j]
                        })

            if correlation_pairs:
                correlation_df = pd.DataFrame(correlation_pairs)
                st.dataframe(correlation_df.sort_values('Corrélation', key=abs, ascending=False))
                
                st.subheader("Recommandations pour gérer la multicolinéarité:")
                st.markdown("""
                Pour les variables fortement corrélées, voici les approches possibles:
                1. **Sélection basée sur l'expertise métier**: Choisir la variable la plus pertinente du point de vue business
                2. **Création de nouvelles caractéristiques**: Combiner les variables corrélées en une nouvelle caractéristique
                3. **Analyse en Composantes Principales (PCA)**: Réduire la dimensionnalité tout en préservant l'information
                4. **Élimination des variables**: Supprimer la variable la moins importante de chaque paire fortement corrélée
                """)

                st.subheader("Actions recommandées:")
                for pair in correlation_pairs:
                    if abs(pair['Corrélation']) > 0.7:
                        st.write(f"- Pour {pair['Variable 1']} et {pair['Variable 2']} (corrélation: {pair['Corrélation']:.2f}):")
                        if pair['Corrélation'] > 0:
                            st.write("  → Considérer de n'en garder qu'une seule ou créer un indice composite")
                        else:
                            st.write("  → Ces variables ont une forte relation inverse, considérer d'utiliser leur ratio")
            else:
                st.info("Aucune corrélation forte trouvée entre les variables")
        else:
            st.info("Pas assez de variables numériques pour l'analyse de corrélation")

        # Analyse de la variable cible
        st.header("6. Analyse de la Variable Cible")
        target_col = st.selectbox("Sélectionnez la variable cible", df.columns)
        
        if target_col:
            if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
                target_dist = df[target_col].value_counts()
                fig = px.pie(values=target_dist.values, 
                           names=target_dist.index, 
                           title=f"Distribution de {target_col}")
                st.plotly_chart(fig)
                
                # Relation avec les variables numériques
                st.subheader("6.1 Relation avec les Variables Numériques")
                numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numerical_cols:
                    if col != target_col:
                        fig = px.box(df, x=target_col, y=col, 
                                   title=f'Distribution de {col} par {target_col}')
                        st.plotly_chart(fig)
            else:
                st.warning("La variable cible sélectionnée n'est pas catégorielle ou a trop de catégories uniques")

        # Bouton pour passer à l'étape suivante
        if st.button("Passer au Preprocessing"):
            st.session_state['current_data'] = df
            st.session_state['target_column'] = target_col
            st.session_state['preprocessing_recommendations'] = preprocessing_recommendations
            st.success("✅ Données sauvegardées! Vous pouvez maintenant lancer le preprocessing.")
            
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du chargement du fichier: {str(e)}")
else:
    st.info("👆 Veuillez uploader un fichier CSV pour commencer l'analyse")

st.header("7. Conclusions et Recommandations")
st.markdown("""
### Préparation des données recommandée:
1. **Traitement des valeurs manquantes**
   - Imputation par la médiane pour les variables numériques
   - Imputation par le mode pour les variables catégorielles

2. **Feature Engineering**
   - Création de ratios pertinents
   - Encodage des variables catégorielles

3. **Normalisation**
   - Standardisation des variables numériques
   - Mise à l'échelle Min-Max pour certaines caractéristiques

4. **Sélection des caractéristiques**
   - Élimination des variables fortement corrélées
   - Sélection basée sur l'importance des caractéristiques

### Approches de Modélisation Suggérées:
1. **Modèles de Base**
   - Régression Logistique (baseline)
   - Random Forest
   - XGBoost

2. **Techniques d'Ensemble**
   - Stacking des meilleurs modèles
   - Voting Classifier

3. **Validation**
   - Validation croisée stratifiée
   - Métriques adaptées aux classes déséquilibrées
""")
