from .base_agent import BaseAgent

class PreprocessingAgent(BaseAgent):
    def analyze(self, df, analysis_recommendations):
        """
        Fournit des recommandations de prétraitement basées sur l'analyse
        """
        # Préparation des statistiques
        dataset_stats = df.describe().to_string()
        
        # Création du prompt
        prompt = f"""
        En tant qu'expert en prétraitement de données pour le risque de crédit, analysez les informations suivantes :

        Statistiques du dataset :
        {dataset_stats}

        Recommandations d'analyse précédentes :
        {analysis_recommendations}

        Veuillez fournir :
        1. Des étapes détaillées de prétraitement à appliquer
        2. La justification de chaque étape
        3. Les transformations spécifiques pour chaque variable
        4. Les points d'attention particuliers

        Réponse :
        """
        
        return self.llm.generate([prompt]).generations[0][0].text
