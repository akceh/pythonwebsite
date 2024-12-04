from .base_agent import BaseAgent

class ModelingAgent(BaseAgent):
    def analyze(self, features_info, model_results, class_distribution):
        """
        Analyse les résultats de la modélisation et fournit des recommandations
        """
        prompt = f"""
        En tant qu'expert en modélisation pour le risque de crédit, analysez les informations suivantes :

        Caractéristiques importantes :
        {features_info}

        Résultats des modèles :
        {model_results}

        Distribution des classes :
        {class_distribution}

        Veuillez fournir :
        1. Une analyse détaillée des performances des modèles
        2. Des recommandations pour l'amélioration des modèles
        3. Des suggestions pour gérer le déséquilibre des classes
        4. Des métriques à surveiller en priorité

        Réponse :
        """
        
        return self.llm.generate([prompt]).generations[0][0].text
