import streamlit as st
from .data_analysis_agent import DataAnalysisAgent
from .preprocessing_agent import PreprocessingAgent
from .modeling_agent import ModelingAgent

class AgentCoordinator:
    def __init__(self, model_path):
        """
        Initialise les agents avec le même modèle LLM
        """
        self.data_analysis_agent = DataAnalysisAgent(model_path)
        self.preprocessing_agent = PreprocessingAgent(model_path)
        self.modeling_agent = ModelingAgent(model_path)

    def get_analysis_recommendations(self, df):
        """
        Obtient les recommandations pour l'analyse des données
        """
        with st.spinner("L'agent analyse vos données..."):
            recommendations = self.data_analysis_agent.analyze(df)
            st.success("✅ Analyse terminée!")
            
            st.subheader("💡 Recommandations de l'Agent d'Analyse")
            st.write(recommendations)
            
            return recommendations

    def get_preprocessing_recommendations(self, df, analysis_recommendations):
        """
        Obtient les recommandations pour le prétraitement
        """
        with st.spinner("L'agent prépare les recommandations de prétraitement..."):
            recommendations = self.preprocessing_agent.analyze(df, analysis_recommendations)
            st.success("✅ Recommandations de prétraitement générées!")
            
            st.subheader("🔧 Recommandations de Prétraitement")
            st.write(recommendations)
            
            return recommendations

    def get_modeling_recommendations(self, features_info, model_results, class_distribution):
        """
        Obtient les recommandations pour la modélisation
        """
        with st.spinner("L'agent analyse les résultats des modèles..."):
            recommendations = self.modeling_agent.analyze(
                features_info, model_results, class_distribution
            )
            st.success("✅ Analyse des modèles terminée!")
            
            st.subheader("🤖 Recommandations de Modélisation")
            st.write(recommendations)
            
            return recommendations
