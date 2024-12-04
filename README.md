# Application de Prédiction de Risque de Crédit

Cette application utilise Streamlit et Machine Learning pour prédire le risque de crédit d'un client.

## Structure du Projet

```
credit_risk_app/
│
├── data/                      # Dossier pour stocker les données
│   └── raw/                  # Données brutes
│
├── notebooks/                 # Notebooks Jupyter pour l'exploration
│
├── src/                      # Code source
│   ├── data_analysis.py      # Analyse exploratoire des données
│   ├── preprocessing.py      # Prétraitement des données
│   ├── modeling.py          # Entraînement des modèles
│   └── app.py               # Application Streamlit principale
│
├── models/                   # Modèles entraînés
│
└── requirements.txt          # Dépendances du projet
```

## Étapes d'Utilisation

1. **Analyse Exploratoire des Données**
   ```bash
   streamlit run src/data_analysis.py
   ```
   - Uploadez votre fichier CSV
   - Analysez les distributions
   - Identifiez les corrélations
   - Choisissez les features pertinentes

2. **Modeling**
   ```bash
   streamlit run src/app.py
   ```
   - Utilisez les insights de l'analyse pour le preprocessing
   - Entraînez et comparez différents modèles
   - Prédisez le risque de crédit

## Installation

1. Clonez le repository
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Notes Importantes

- Commencez toujours par l'analyse exploratoire (`data_analysis.py`) avant le modeling
- Les visualisations sont interactives (zoom, filtres, etc.)
- Les modèles sont automatiquement sauvegardés dans le dossier `models/`
