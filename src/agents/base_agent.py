from abc import ABC, abstractmethod
from langchain_community.llms import GPT4All
import os
import psutil
import gc
import time

class BaseAgent(ABC):
    def __init__(self, model_path):
        """
        Initialise l'agent avec un modèle LLM
        """
        if not os.path.exists(model_path):
            raise ValueError(f"Le modèle {model_path} n'existe pas!")

        # Force garbage collection
        gc.collect()
        
        # Vérifier la mémoire disponible
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # En GB
        if available_memory < 2:
            raise MemoryError(f"Mémoire insuffisante: {available_memory:.2f}GB disponible. Minimum requis: 2GB")

        # Attendre un peu pour laisser le système se stabiliser
        time.sleep(2)

        try:
            model_kwargs = {
                'n_threads': 1,
                'n_ctx': 32,
                'n_batch': 1,
                'top_k': 1,
                'top_p': 0.95,
                'temp': 0.7,
                'repeat_penalty': 1.1,
                'max_tokens': 32,
            }

            self.llm = GPT4All(
                model=model_path,
                backend='gptj',  # Utiliser gptj qui est plus stable
                verbose=True,
                allow_download=False,
                **model_kwargs
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "cuda" in error_msg or "gpu" in error_msg:
                raise RuntimeError("Erreur GPU détectée. Désactivez CUDA dans gpt4all ou réinstallez sans support CUDA.")
            elif "memory" in error_msg or "allocation" in error_msg:
                raise MemoryError(f"Mémoire insuffisante. Disponible: {available_memory:.2f}GB")
            elif "load" in error_msg:
                raise RuntimeError("Erreur de chargement du modèle. Vérifiez que le fichier n'est pas corrompu.")
            else:
                raise RuntimeError(f"Erreur inattendue: {str(e)}")

    def _ensure_memory_available(self):
        """Vérifie qu'il y a assez de mémoire disponible"""
        gc.collect()  # Force garbage collection
        available = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        if available < 0.5:  # Moins de 500MB disponible
            raise MemoryError(f"Mémoire critique: {available:.2f}GB")
        return available

    @abstractmethod
    def analyze(self, *args, **kwargs):
        """
        Analyse les données et retourne des recommandations
        """
        try:
            self._ensure_memory_available()
            # La méthode réelle sera implémentée dans les classes enfants
            pass
        except Exception as e:
            raise RuntimeError(f"Erreur durant l'analyse: {str(e)}")
