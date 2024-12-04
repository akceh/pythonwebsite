import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# Configuration de la page - doit être la première commande Streamlit
st.set_page_config(
    page_title="Credit Risk Analysis App",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    /* Style général */
    .stApp {
        background-color: #f5f7f9;
    }
    
    /* Style des titres */
    h1 {
        color: #1e3d59;
        padding: 20px 0;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        border-bottom: 2px solid #17a2b8;
        margin-bottom: 30px;
    }
    
    h2 {
        color: #2c5282;
        margin-top: 30px;
        font-size: 1.8em;
    }
    
    /* Style des boutons */
    .stButton > button {
        background-color: #17a2b8;
        color: white;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #138496;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Style des cartes d'information */
    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin: 10px 0;
    }
    
    /* Style des alertes */
    .stAlert {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Style des widgets */
    .stSelectbox, .stMultiSelect {
        background-color: white;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Style des séparateurs */
    hr {
        margin: 30px 0;
        border: none;
        border-top: 2px solid #e2e8f0;
    }
    
    /* Style des DataFrames */
    .dataframe {
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05) !important;
    }
    
    /* Style des recommandations */
    .recommendation-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 15px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    
    /* Style des étapes de progression */
    .step-progress {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 30px 0;
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    
    .step {
        text-align: center;
        color: #718096;
    }
    
    .step.active {
        color: #17a2b8;
        font-weight: bold;
    }
    
    /* Style des métriques */
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #17a2b8;
    }
    
    .metric-label {
        color: #718096;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# Initialisation des variables de session si elles n'existent pas
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'data_analysis'
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False
if 'modeling_done' not in st.session_state:
    st.session_state.modeling_done = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

def get_analysis_recommendations(df):
    """Generate basic data analysis recommendations."""
    recommendations = []
    
    # Check for missing values
    missing_vals = df.isnull().sum()
    if missing_vals.any():
        recommendations.append(f"Il y a {missing_vals.sum()} valeurs manquantes dans le dataset.")
    
    # Check for numerical and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    recommendations.append(f"Le dataset contient {len(num_cols)} colonnes numériques et {len(cat_cols)} colonnes catégorielles.")
    
    # Basic statistics
    recommendations.append("Statistiques de base pour les colonnes numériques:")
    for col in num_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        recommendations.append(f"- {col}: moyenne = {mean_val:.2f}, écart-type = {std_val:.2f}")
    
    return recommendations

def get_preprocessing_recommendations(df):
    """Generate sophisticated preprocessing recommendations based on data analysis."""
    recommendations = []
    
    # 1. Analyse des valeurs manquantes
    missing_vals = df.isnull().sum()
    missing_percentages = (missing_vals / len(df)) * 100
    
    if missing_vals.any():
        recommendations.append(" Traitement des valeurs manquantes recommandé:")
        for col in missing_vals[missing_vals > 0].index:
            percentage = missing_percentages[col]
            if percentage > 50:
                recommendations.append(f"-  Colonne '{col}': {missing_vals[col]} valeurs manquantes ({percentage:.1f}%) - Considérer la suppression de cette colonne")
            elif percentage > 30:
                recommendations.append(f"-  Colonne '{col}': {missing_vals[col]} valeurs manquantes ({percentage:.1f}%) - Utiliser des techniques avancées d'imputation (KNN ou MICE)")
            else:
                if df[col].dtype in ['int64', 'float64']:
                    recommendations.append(f"-  Colonne '{col}': {missing_vals[col]} valeurs manquantes ({percentage:.1f}%) - Utiliser la médiane/moyenne")
                else:
                    recommendations.append(f"-  Colonne '{col}': {missing_vals[col]} valeurs manquantes ({percentage:.1f}%) - Utiliser le mode")
    
    # 2. Analyse des variables catégorielles
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        recommendations.append("\n Encodage des variables catégorielles recommandé:")
        for col in cat_cols:
            unique_vals = df[col].nunique()
            unique_ratio = unique_vals / len(df)
            
            if unique_ratio > 0.5:
                recommendations.append(f"-  '{col}': Haute cardinalité ({unique_vals} valeurs uniques, {unique_ratio:.1%}) - Considérer le regroupement ou la suppression")
            elif unique_vals < 10:
                value_counts = df[col].value_counts()
                imbalance = value_counts.max() / value_counts.min()
                if imbalance > 10:
                    recommendations.append(f"-  '{col}': One-Hot Encoding avec attention aux classes déséquilibrées (ratio {imbalance:.1f}:1)")
                else:
                    recommendations.append(f"-  '{col}': One-Hot Encoding recommandé ({unique_vals} valeurs uniques)")
            else:
                recommendations.append(f"-  '{col}': Label Encoding ou Target Encoding ({unique_vals} valeurs uniques)")
    
    # 3. Analyse des variables numériques
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 0:
        recommendations.append("\n Analyse des variables numériques:")
        for col in num_cols:
            # Calcul des statistiques
            skewness = df[col].skew()
            std_dev = df[col].std()
            mean = df[col].mean()
            cv = std_dev / mean if mean != 0 else float('inf')
            
            # Détection des outliers avec IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            
            if abs(skewness) > 1:
                recommendations.append(f"-  '{col}': Distribution asymétrique (skewness={skewness:.2f}) - Considérer une transformation log ou Box-Cox")
            
            if cv > 1:
                recommendations.append(f"-  '{col}': Grande variabilité (CV={cv:.2f}) - Normalisation recommandée")
            
            if outliers > 0:
                outlier_percentage = (outliers / len(df)) * 100
                recommendations.append(f"-  '{col}': {outliers} outliers détectés ({outlier_percentage:.1f}%) - Considérer le capping ou la transformation")
    
    # 4. Analyse des corrélations
    if len(num_cols) > 1:
        corr_matrix = df[num_cols].corr()
        high_corr_pairs = []
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                if abs(corr_matrix.iloc[i,j]) > 0.8:
                    high_corr_pairs.append((num_cols[i], num_cols[j], corr_matrix.iloc[i,j]))
        
        if high_corr_pairs:
            recommendations.append("\n Analyse des corrélations:")
            for col1, col2, corr in high_corr_pairs:
                recommendations.append(f"-  Forte corrélation entre '{col1}' et '{col2}' ({corr:.2f}) - Considérer la suppression d'une variable")
    
    return recommendations

def get_modeling_recommendations(df):
    """Generate modeling recommendations."""
    recommendations = [
        "Recommandations pour la modélisation:",
        "1. Algorithmes recommandés:",
        "   - Random Forest (bon pour gérer les features non-linéaires)",
        "   - Gradient Boosting (performances élevées)",
        "   - Régression Logistique (bonne interprétabilité)",
        "",
        "2. Validation croisée:",
        "   - Utiliser k-fold cross validation (k=5)",
        "   - Stratifier les folds pour gérer le déséquilibre des classes",
        "",
        "3. Métriques d'évaluation:",
        "   - Accuracy pour la performance globale",
        "   - Precision et Recall pour les faux positifs/négatifs",
        "   - F1-score pour l'équilibre precision/recall",
        "   - ROC-AUC pour la discrimination du modèle"
    ]
    return recommendations

def data_analysis():
    st.markdown("<h1>🔍 Analyse des Données</h1>", unsafe_allow_html=True)
    
    # Section de téléchargement des données
    st.markdown("<h2>📤 Chargement des Données</h2>", unsafe_allow_html=True)
    with st.container():
        st.markdown(
            """
            <div class="info-card">
                <h3>Importez votre fichier de données</h3>
                <p>Formats supportés : CSV, Excel</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        uploaded_file = st.file_uploader("Choisir un fichier", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.current_data = df
                st.session_state.data_uploaded = True
                
                # Affichage des informations sur le dataset
                st.markdown("<h2>📊 Aperçu des Données</h2>", unsafe_allow_html=True)
                
                # Métriques principales
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{df.shape[0]}</div>
                            <div class="metric-label">Observations</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{df.shape[1]}</div>
                            <div class="metric-label">Variables</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col3:
                    num_cols = len(df.select_dtypes(include=['int64', 'float64']).columns)
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{num_cols}</div>
                            <div class="metric-label">Variables Numériques</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col4:
                    cat_cols = len(df.select_dtypes(include=['object', 'category']).columns)
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{cat_cols}</div>
                            <div class="metric-label">Variables Catégorielles</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Aperçu des données
                with st.container():
                    st.markdown(
                        """
                        <div class="info-card">
                            <h3>Aperçu du Dataset</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.dataframe(df.head())
                
                # Sélection de la variable cible
                st.markdown("<h2>🎯 Sélection de la Variable Cible</h2>", unsafe_allow_html=True)
                target_column = st.selectbox(
                    "Choisissez la variable à prédire",
                    df.columns.tolist()
                )
                st.session_state.target_column = target_column
                
                # Analyse des données
                st.markdown("<h2>📈 Analyse Statistique</h2>", unsafe_allow_html=True)
                
                # Description statistique
                with st.container():
                    st.markdown(
                        """
                        <div class="info-card">
                            <h3>Statistiques Descriptives</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.dataframe(df.describe())
                
                # Types des variables
                with st.container():
                    st.markdown(
                        """
                        <div class="info-card">
                            <h3>Types des Variables</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    dtypes_df = pd.DataFrame(df.dtypes, columns=['Type'])
                    st.dataframe(dtypes_df)
                
                # Valeurs manquantes
                missing_values = df.isnull().sum()
                if missing_values.any():
                    st.markdown("<h2>⚠️ Valeurs Manquantes</h2>", unsafe_allow_html=True)
                    missing_df = pd.DataFrame({
                        'Variable': missing_values.index,
                        'Nombre': missing_values.values,
                        'Pourcentage': (missing_values / len(df) * 100).round(2)
                    })
                    missing_df = missing_df[missing_df['Nombre'] > 0]
                    st.dataframe(missing_df)
                
                # Bouton pour passer au prétraitement
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Passer au Prétraitement", type="primary"):
                        st.session_state.current_step = 'preprocessing'
                        st.rerun()
                
            except Exception as e:
                st.error(f"Une erreur s'est produite lors du chargement du fichier : {str(e)}")
        
        else:
            st.info("👆 Veuillez télécharger un fichier pour commencer l'analyse")

def preprocessing():
    st.markdown(
        """
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: #1E88E5; font-size: 2.5em;'>🔄 Prétraitement des Données</h1>
            <p style='color: #666; font-size: 1.2em;'>Optimisez vos données pour l'analyse</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    if not st.session_state.data_uploaded:
        st.error("⚠️ Veuillez d'abord télécharger et analyser les données.")
        return
    
    try:
        # Récupération des données
        df = st.session_state.get('current_data')
        target_column = st.session_state.get('target_column')
        
        if df is None or target_column is None:
            st.error("⚠️ Les données ou la colonne cible ne sont pas disponibles.")
            return
        
        # Création de métriques clés
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-value'>{df.shape[0]}</div>
                    <div class='metric-label'>Observations</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100).round(2)
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-value'>{missing_percentage}%</div>
                    <div class='metric-label'>Valeurs Manquantes</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col3:
            cat_cols = len(df.select_dtypes(include=['object', 'category']).columns)
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-value'>{cat_cols}</div>
                    <div class='metric-label'>Variables Catégorielles</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col4:
            num_cols = len(df.select_dtypes(include=['int64', 'float64']).columns)
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-value'>{num_cols}</div>
                    <div class='metric-label'>Variables Numériques</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Visualisation des données manquantes
        st.markdown("<h2>📊 Analyse des Valeurs Manquantes</h2>", unsafe_allow_html=True)
        missing_df = pd.DataFrame({
            'Variable': df.columns,
            'Pourcentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        fig = px.bar(
            missing_df,
            x='Variable',
            y='Pourcentage',
            title='Pourcentage de Valeurs Manquantes par Variable',
            color='Pourcentage',
            color_continuous_scale='RdYlBu_r'
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Génération et affichage des recommandations
        preprocessing_recommendations = get_preprocessing_recommendations(df)
        
        st.markdown(
            """
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <h2 style='color: #2c5282;'>🎯 Recommandations de Prétraitement</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Organiser les recommandations par catégorie
        categories = {
            "Valeurs Manquantes": [],
            "Variables Catégorielles": [],
            "Variables Numériques": [],
            "Corrélations": []
        }

        for rec in preprocessing_recommendations:
            if "valeurs manquantes" in rec.lower():
                categories["Valeurs Manquantes"].append(rec)
            elif "catégorielle" in rec.lower():
                categories["Variables Catégorielles"].append(rec)
            elif "numérique" in rec.lower() or "distribution" in rec.lower():
                categories["Variables Numériques"].append(rec)
            elif "corrélation" in rec.lower():
                categories["Corrélations"].append(rec)

        # Afficher les recommandations par catégorie avec des icônes
        icons = {
            "Valeurs Manquantes": "❌",
            "Variables Catégorielles": "📝",
            "Variables Numériques": "📊",
            "Corrélations": "🔗"
        }

        for category, recs in categories.items():
            if recs:
                st.markdown(
                    f"""
                    <div class='recommendation-category'>
                        <h3>{icons[category]} {category}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                for rec in recs:
                    st.markdown(
                        f"""
                        <div class='recommendation-box'>
                            {rec}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # Options de prétraitement
        st.markdown("<h2>⚙️ Configuration du Prétraitement</h2>", unsafe_allow_html=True)
        
        tabs = st.tabs(["🔍 Valeurs Manquantes", "🔄 Encodage", "📊 Normalisation"])
        
        with tabs[0]:
            handle_missing = st.checkbox("Traiter les valeurs manquantes", value=True)
            if handle_missing:
                col1, col2 = st.columns(2)
                with col1:
                    numerical_strategy = st.selectbox(
                        "Stratégie pour les variables numériques",
                        ["mean", "median", "most_frequent", "constant"],
                        format_func=lambda x: {
                            "mean": "Moyenne",
                            "median": "Médiane",
                            "most_frequent": "Mode",
                            "constant": "Valeur constante"
                        }[x]
                    )
                with col2:
                    categorical_strategy = st.selectbox(
                        "Stratégie pour les variables catégorielles",
                        ["most_frequent", "constant"],
                        format_func=lambda x: {
                            "most_frequent": "Mode",
                            "constant": "Valeur constante"
                        }[x]
                    )

        with tabs[1]:
            encode_categorical = st.checkbox("Encoder les variables catégorielles", value=True)
            if encode_categorical:
                encoding_strategy = st.selectbox(
                    "Méthode d'encodage",
                    ["label", "onehot"],
                    format_func=lambda x: {
                        "label": "Label Encoding",
                        "onehot": "One-Hot Encoding"
                    }[x]
                )

        with tabs[2]:
            scale_features = st.checkbox("Normaliser les variables", value=True)
            if scale_features:
                st.info("📝 Les variables numériques seront normalisées avec la méthode Z-score (moyenne=0, écart-type=1)")

        # Sélection des caractéristiques
        st.markdown("<h2>🎯 Sélection des Variables</h2>", unsafe_allow_html=True)
        selected_features = st.multiselect(
            "Sélectionnez les variables à inclure dans le modèle",
            [col for col in df.columns if col != target_column],
            default=[col for col in df.columns if col != target_column]
        )

        # Bouton de prétraitement
        if st.button("🚀 Lancer le Prétraitement", type="primary"):
            with st.spinner("Prétraitement en cours..."):
                try:
                    # Code de prétraitement existant...
                    # Étape 1 : Encodage des variables catégorielles
                    df_encoded = df.copy()
                    if encode_categorical:
                        with st.spinner("Encodage des variables catégorielles en cours..."):
                            st.info("Encodage des variables catégorielles...")
                            if encoding_strategy == "label":
                                for col in selected_features:
                                    if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
                                        df_encoded[col] = df_encoded[col].astype('category').cat.codes
                            elif encoding_strategy == "onehot":
                                df_encoded = pd.get_dummies(df_encoded, columns=[
                                    col for col in selected_features if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category'
                                ], drop_first=True)
                            st.success(" Encodage terminé avec succès!")
                            st.write("Aperçu après encodage:")
                            st.dataframe(df_encoded.head())

                    # Étape 2 : Traitement des valeurs manquantes
                    if handle_missing:
                        with st.spinner("Traitement des valeurs manquantes en cours..."):
                            st.info("Traitement des valeurs manquantes...")
                            for col in df_encoded.columns:
                                if col != target_column:  # Ne pas traiter la colonne cible
                                    if df_encoded[col].dtype in ['float64', 'int64']:
                                        if numerical_strategy == "median":
                                            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
                                        elif numerical_strategy == "mean":
                                            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].mean())
                                        elif numerical_strategy == "most_frequent":
                                            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].mode()[0])
                                        elif numerical_strategy == "constant":
                                            df_encoded[col] = df_encoded[col].fillna(0)
                                    else:
                                        if categorical_strategy == "most_frequent":
                                            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].mode()[0])
                                        elif categorical_strategy == "constant":
                                            df_encoded[col] = df_encoded[col].fillna("missing")
                            st.success(" Traitement des valeurs manquantes terminé!")
                            st.write("Aperçu après traitement des valeurs manquantes:")
                            st.dataframe(df_encoded.head())

                    # Étape 3 : Normalisation
                    if scale_features:
                        with st.spinner("Normalisation des variables numériques en cours..."):
                            st.info("Normalisation des variables numériques...")
                            for col in df_encoded.columns:
                                if col != target_column and df_encoded[col].dtype in ['float64', 'int64']:
                                    df_encoded[col] = (df_encoded[col] - df_encoded[col].mean()) / df_encoded[col].std()
                            st.success(" Normalisation terminée!")
                            st.write("Aperçu après normalisation:")
                            st.dataframe(df_encoded.head())

                    # Sauvegarde des données prétraitées
                    st.session_state['processed_data'] = df_encoded
                    st.session_state.preprocessing_done = True
                    st.success(" Prétraitement terminé avec succès!")

                    # Affichage des résultats finaux
                    st.subheader("Aperçu des Données Prétraitées")
                    st.dataframe(df_encoded.head())
                    st.write("Forme des données :", df_encoded.shape)
                    st.write("Types de données :", df_encoded.dtypes)

                except Exception as e:
                    st.error(f"Une erreur s'est produite lors du prétraitement : {str(e)}")
                    return

        # Navigation vers le modeling (en dehors du bloc try-except)
        if st.session_state.preprocessing_done:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(" Passer au Modeling", key="goto_modeling"):
                    st.session_state.current_step = 'modeling'
                    st.rerun()

    except Exception as e:
        st.error(f"Une erreur s'est produite lors de l'importation ou de l'exécution : {str(e)}")


def modeling():
    st.title(" Modeling")
    st.markdown("---")
    
    if not st.session_state.preprocessing_done:
        st.error(" Veuillez d'abord prétraiter les données")
        return
    
    df = st.session_state.processed_data
    target_column = st.session_state.target_column
    
    # Recommandations de l'agent pour le modeling
    with st.spinner(" L'agent analyse les options de modeling..."):
        modeling_recommendations = get_modeling_recommendations(df)
    
    # Interface en colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Configuration du Modeling")
        test_size = st.slider(
            "Taille du jeu de test (%)",
            10, 40, 20,
            help="Pourcentage des données à utiliser pour le test"
        )
        
        models_to_train = st.multiselect(
            "Sélectionnez les modèles à entraîner",
            ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM"],
            default=["Logistic Regression", "Random Forest"],
            help="Choisissez un ou plusieurs modèles à comparer"
        )
    
    with col2:
        st.subheader(" Recommandations de l'Agent")
        st.markdown(f"""
        <div class="success-message">
        {modeling_recommendations}
        </div>
        """, unsafe_allow_html=True)
    
    if st.button(" Lancer l'Entraînement"):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )
        
        results = {}
        
        for model_name in models_to_train:
            with st.spinner(f" Entraînement du modèle {model_name}..."):
                if model_name == "Logistic Regression":
                    model = LogisticRegression()
                elif model_name == "Random Forest":
                    model = RandomForestClassifier()
                elif model_name == "Gradient Boosting":
                    model = GradientBoostingClassifier()
                else:  # SVM
                    model = SVC(probability=True)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                results[model_name] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred)
                }
        
        # Affichage des résultats avec graphiques
        st.header(" Résultats")
        
        # Graphique comparatif
        metrics_df = pd.DataFrame(results).T
        fig = go.Figure()
        for metric in metrics_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=metrics_df.index,
                y=metrics_df[metric],
                text=metrics_df[metric].round(3)
            ))
        
        fig.update_layout(
            title="Comparaison des Performances des Modèles",
            barmode='group',
            yaxis_title="Score",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau détaillé
        st.subheader(" Détails des Métriques")
        st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)
        
        # Sauvegarde du meilleur modèle
        best_model_name = metrics_df['f1'].idxmax()
        st.success(f" Meilleur modèle : {best_model_name} (F1-Score: {metrics_df.loc[best_model_name, 'f1']:.3f})")
        
        st.session_state.modeling_done = True

def main():
    # Initialisation des variables de session
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'data_analysis'
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'preprocessing_done' not in st.session_state:
        st.session_state.preprocessing_done = False
    if 'modeling_done' not in st.session_state:
        st.session_state.modeling_done = False
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    # Barre de progression des étapes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""<div class="step {'active' if st.session_state.current_step == 'data_analysis' else ''}">
                1. 📊 Analyse des Données
            </div>""",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""<div class="step {'active' if st.session_state.current_step == 'preprocessing' else ''}">
                2. 🔄 Prétraitement
            </div>""",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""<div class="step {'active' if st.session_state.current_step == 'modeling' else ''}">
                3. 🎯 Modélisation
            </div>""",
            unsafe_allow_html=True
        )

    # Navigation principale
    if st.session_state.current_step == 'data_analysis':
        data_analysis()
    elif st.session_state.current_step == 'preprocessing':
        preprocessing()
    elif st.session_state.current_step == 'modeling':
        modeling()
    else:
        st.error("Étape inconnue. Veuillez redémarrer l'application.")



if __name__ == "__main__":
    main()
