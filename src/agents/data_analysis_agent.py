from .base_agent import BaseAgent

class DataAnalysisAgent(BaseAgent):
    def analyze(self, df):
        """
        Analyse le dataset et fournit des recommandations
        """
        # Préparation des statistiques
        dataset_stats = df.describe().to_string()
        distributions = df.dtypes.to_string()
        correlations = df.corr().to_string()
        
        # Création du prompt
        prompt = f"""
        En tant qu'expert en analyse de données pour le risque de crédit, analysez les informations suivantes :

        Statistiques du dataset :
        {dataset_stats}

        Distribution des variables :
        {distributions}

        Corrélations :
        {correlations}

        Veuillez fournir :
        1. Une analyse approfondie des patterns observés
        2. Des recommandations pour le prétraitement des données
        3. Les variables qui semblent les plus importantes pour la prédiction
        4. Les potentiels problèmes à adresser

        Réponse :
        """
        
        return self.llm.generate([prompt]).generations[0][0].text
