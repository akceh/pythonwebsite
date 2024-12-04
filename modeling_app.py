import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import joblib
from src.agents.agent_coordinator import AgentCoordinator

# Configuration de la page
st.set_page_config(page_title="Modeling - Prédiction de Risque de Crédit", layout="wide")
st.title("Modeling - Prédiction de Risque de Crédit")

# Vérification des données prétraitées
if 'processed_data' not in st.session_state or 'preprocessor' not in st.session_state:
    st.error("❌ Données non prétraitées. Veuillez d'abord exécuter preprocessing_app.py")
    st.stop()

# Récupération des données
df = st.session_state['processed_data']
preprocessor = st.session_state['preprocessor']
target_column = st.session_state['target_column']

# Séparation features/target
X = df.drop(columns=[target_column])
y = df[target_column]

# Configuration du modeling
st.header("Configuration du Modeling")

# Paramètres de split des données
test_size = st.slider("Taille du jeu de test (%)", 10, 40, 20) / 100
random_state = st.number_input("Random State", 0, 100, 42)

# Sélection des modèles
st.subheader("Sélection des Modèles")
models_to_train = st.multiselect(
    "Choisissez les modèles à entraîner",
    ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM"],
    default=["Logistic Regression", "Random Forest"]
)

# Configuration des hyperparamètres
st.subheader("Configuration des Hyperparamètres")

model_params = {}
if "Logistic Regression" in models_to_train:
    st.write("Logistic Regression:")
    model_params["lr_C"] = st.slider("C (Régularisation)", 0.01, 10.0, 1.0)

if "Random Forest" in models_to_train:
    st.write("Random Forest:")
    model_params["rf_n_estimators"] = st.slider("Nombre d'arbres", 10, 200, 100)
    model_params["rf_max_depth"] = st.slider("Profondeur maximale", 2, 20, 10)

if "Gradient Boosting" in models_to_train:
    st.write("Gradient Boosting:")
    model_params["gb_n_estimators"] = st.slider("Nombre d'estimateurs", 10, 200, 100)
    model_params["gb_learning_rate"] = st.slider("Learning Rate", 0.01, 0.3, 0.1)

if "SVM" in models_to_train:
    st.write("SVM:")
    model_params["svm_C"] = st.slider("C (SVM)", 0.1, 10.0, 1.0)

# Bouton pour lancer l'entraînement
if st.button("Lancer l'Entraînement"):
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Dictionnaire pour stocker les résultats
    results = {}
    trained_models = {}
    
    # Entraînement des modèles
    for model_name in models_to_train:
        with st.spinner(f"Entraînement du modèle {model_name}..."):
            if model_name == "Logistic Regression":
                model = LogisticRegression(C=model_params["lr_C"])
            elif model_name == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=model_params["rf_n_estimators"],
                    max_depth=model_params["rf_max_depth"]
                )
            elif model_name == "Gradient Boosting":
                model = GradientBoostingClassifier(
                    n_estimators=model_params["gb_n_estimators"],
                    learning_rate=model_params["gb_learning_rate"]
                )
            else:  # SVM
                model = SVC(C=model_params["svm_C"], probability=True)
            
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test)
            
            # Métriques
            results[model_name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred)
            }
            
            # Sauvegarde du modèle
            trained_models[model_name] = model
    
    # Affichage des résultats
    st.header("Résultats")
    for model_name, metrics in results.items():
        st.subheader(model_name)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        col2.metric("Precision", f"{metrics['precision']:.3f}")
        col3.metric("Recall", f"{metrics['recall']:.3f}")
        col4.metric("F1 Score", f"{metrics['f1']:.3f}")
    
    # Sélection du meilleur modèle
    best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
    best_model = trained_models[best_model_name]
    
    # Sauvegarde du meilleur modèle
    joblib.dump(best_model, 'models/best_model.joblib')
    st.success(f"✅ Meilleur modèle ({best_model_name}) sauvegardé!")
    
    # Obtenir les recommandations de l'agent
    agent_coordinator = AgentCoordinator("models/ggml-gpt4all-j-v1.3-groovy.bin")
    modeling_recommendations = agent_coordinator.get_modeling_recommendations(
        str(X.columns.tolist()),
        str(results),
        str(pd.Series(y).value_counts())
    )
    
    st.header("Recommandations de l'Agent")
    st.write(modeling_recommendations)

# Instructions d'utilisation
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Configurez les paramètres de split des données
2. Sélectionnez les modèles à entraîner
3. Ajustez les hyperparamètres
4. Lancez l'entraînement
5. Analysez les résultats
6. Consultez les recommandations de l'agent
""")
