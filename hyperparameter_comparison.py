"""
SYSTÈME DE COMPARAISON D'HYPERPARAMÈTRES
Permet de comparer l'impact de différents hyperparamètres sur un même algorithme

Features:
- Définition facile de configurations
- Exécution automatique de multiples expériences
- Visualisations professionnelles
- Statistiques détaillées
- Export des résultats

Auteur: Smart City RL Framework
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime
import pandas as pd

# Import de l'environnement
import sys
sys.path.append('.')
from smartcity_rl_balanced import (
    SmartCityResourceEnv, 
    QLearningSmartCity, 
    DoubleDQNAgent,
    PPOCallback
)


# ============================================================================
# CLASSE DE CONFIGURATION D'EXPÉRIENCE
# ============================================================================

@dataclass
class ExperimentConfig:
    """
    Configuration d'une expérience
    Permet de définir tous les hyperparamètres et paramètres d'entraînement
    """
    name: str  # Nom de l'expérience
    algorithm: str  # 'qlearning', 'double_dqn', 'ppo'
    
    # Hyperparamètres spécifiques
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    
    # Paramètres d'entraînement
    n_episodes: int = 5000  # Pour Q-Learning et DQN
    total_timesteps: int = 2500000  # Pour PPO
    
    # Paramètres d'environnement
    env_params: Dict[str, Any] = field(default_factory=lambda: {
        'num_users': 200,
        'max_steps': 500
    })
    
    # Métadonnées
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self):
        """Convertit la config en dictionnaire"""
        return {
            'name': self.name,
            'algorithm': self.algorithm,
            'hyperparams': self.hyperparams,
            'n_episodes': self.n_episodes,
            'total_timesteps': self.total_timesteps,
            'env_params': self.env_params,
            'description': self.description,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data):
        """Crée une config depuis un dictionnaire"""
        return cls(**data)


@dataclass
class ExperimentResult:
    """Résultat d'une expérience"""
    config: ExperimentConfig
    episode_rewards: List[float]
    success_rates: List[float]
    avg_latencies: List[float]
    training_time: float
    final_metrics: Dict[str, float]
    agent: Any = None


# ============================================================================
# GESTIONNAIRE D'EXPÉRIENCES
# ============================================================================

class ExperimentRunner:
    """
    Gestionnaire d'expériences pour comparaison d'hyperparamètres
    """
    
    def __init__(self, save_dir='./experiments'):
        self.save_dir = save_dir
        self.results: List[ExperimentResult] = []
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Exécute une expérience selon la configuration
        """
        print("\n" + "="*80)
        print(f"🔬 EXPÉRIENCE: {config.name}")
        print(f"   Algorithme: {config.algorithm.upper()}")
        print(f"   Description: {config.description}")
        print("="*80 + "\n")
        
        start_time = datetime.now()
        
        # Exécution selon l'algorithme
        if config.algorithm == 'qlearning':
            agent, rewards, success, latency = self._run_qlearning(config)
        elif config.algorithm == 'double_dqn':
            agent, rewards, success, latency = self._run_double_dqn(config)
        elif config.algorithm == 'ppo':
            agent, rewards, success, latency = self._run_ppo(config)
        else:
            raise ValueError(f"Algorithme inconnu: {config.algorithm}")
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Calcul des métriques finales
        final_metrics = {
            'mean_reward': float(np.mean(rewards[-100:])),
            'std_reward': float(np.std(rewards[-100:])),
            'mean_success_rate': float(np.mean(success[-100:])),
            'mean_latency': float(np.mean(latency[-100:])),
            'variance_reward': float(np.var(rewards[-100:])),
            'min_reward': float(np.min(rewards[-100:])),
            'max_reward': float(np.max(rewards[-100:]))
        }
        
        result = ExperimentResult(
            config=config,
            episode_rewards=rewards,
            success_rates=success,
            avg_latencies=latency,
            training_time=training_time,
            final_metrics=final_metrics,
            agent=agent
        )
        
        self.results.append(result)
        
        print(f"\n✅ Expérience terminée en {training_time:.1f}s")
        print(f"   Reward final: {final_metrics['mean_reward']:.2f} ± {final_metrics['std_reward']:.2f}")
        print(f"   Taux succès: {final_metrics['mean_success_rate']:.1f}%")
        
        return result
    
    def _run_qlearning(self, config: ExperimentConfig) -> Tuple:
        """Entraîne Q-Learning avec la config donnée"""
        env = SmartCityResourceEnv(**config.env_params)
        
        # Créer agent avec hyperparamètres personnalisés
        agent = QLearningSmartCity()
        for param, value in config.hyperparams.items():
            setattr(agent, param, value)
        
        episode_rewards = []
        success_rates = []
        avg_latencies = []
        
        for episode in range(config.n_episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = agent.choose_action(obs, training=True)
                next_obs, reward, done, truncated, info = env.step(action)
                agent.update(obs, action, reward, next_obs, done)
                total_reward += reward
                obs = next_obs
            
            agent.decay_epsilon()
            episode_rewards.append(total_reward)
            
            success_rate = env.successful_tasks / (env.successful_tasks + env.failed_tasks) * 100
            success_rates.append(success_rate)
            
            if env.successful_tasks > 0:
                avg_latencies.append(env.total_latency / env.successful_tasks)
            
            if episode % 500 == 0:
                print(f"  Episode {episode}/{config.n_episodes} | "
                      f"Reward: {np.mean(episode_rewards[-100:]):.2f} | "
                      f"Success: {np.mean(success_rates[-100:]):.1f}%")
        
        return agent, episode_rewards, success_rates, avg_latencies
    
    def _run_double_dqn(self, config: ExperimentConfig) -> Tuple:
        """Entraîne Double DQN avec la config donnée"""
        env = SmartCityResourceEnv(**config.env_params)
        
        # Créer agent avec hyperparamètres personnalisés
        agent = DoubleDQNAgent()
        for param, value in config.hyperparams.items():
            if hasattr(agent, param):
                setattr(agent, param, value)
            # Gérer learning rate spécialement
            if param == 'learning_rate':
                agent.optimizer = optim.Adam(agent.q_network.parameters(), lr=value)
        
        episode_rewards = []
        success_rates = []
        avg_latencies = []
        
        for episode in range(config.n_episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = agent.choose_action(obs, training=True)
                next_obs, reward, done, truncated, info = env.step(action)
                agent.remember(obs, action, reward, next_obs, done)
                agent.replay()
                total_reward += reward
                obs = next_obs
            
            agent.decay_epsilon()
            episode_rewards.append(total_reward)
            
            success_rate = env.successful_tasks / (env.successful_tasks + env.failed_tasks) * 100
            success_rates.append(success_rate)
            
            if env.successful_tasks > 0:
                avg_latencies.append(env.total_latency / env.successful_tasks)
            
            if episode % 500 == 0:
                print(f"  Episode {episode}/{config.n_episodes} | "
                      f"Reward: {np.mean(episode_rewards[-100:]):.2f} | "
                      f"Success: {np.mean(success_rates[-100:]):.1f}%")
        
        return agent, episode_rewards, success_rates, avg_latencies
    
    def _run_ppo(self, config: ExperimentConfig) -> Tuple:
        """Entraîne PPO avec la config donnée"""
        env = make_vec_env(
            lambda: SmartCityResourceEnv(**config.env_params), 
            n_envs=1
        )
        
        # Créer modèle PPO avec hyperparamètres personnalisés
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            **config.hyperparams
        )
        
        callback = PPOCallback()
        model.learn(total_timesteps=config.total_timesteps, callback=callback)
        
        return model, callback.episode_rewards, callback.success_rates, callback.avg_latencies
    
    def compare_results(self, save_plots=True):
        """
        Compare tous les résultats et génère des visualisations
        """
        if len(self.results) < 2:
            print("⚠️ Au moins 2 expériences nécessaires pour comparaison")
            return
        
        print("\n" + "="*80)
        print(f"📊 COMPARAISON DE {len(self.results)} EXPÉRIENCES")
        print("="*80 + "\n")
        
        self._plot_learning_curves()
        self._plot_hyperparameter_sensitivity()
        self._plot_performance_comparison()
        self._plot_stability_analysis()
        self._generate_summary_table()
        
        if save_plots:
            print("\n✅ Tous les graphiques ont été sauvegardés")
    
    def _plot_learning_curves(self):
        """Graphique 1: Courbes d'apprentissage superposées"""
        plt.figure(figsize=(14, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        window = 50
        
        for idx, result in enumerate(self.results):
            rewards = result.episode_rewards
            if len(rewards) > window:
                smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
                plt.plot(smooth, color=colors[idx], linewidth=2.5, 
                        label=result.config.name, alpha=0.85)
        
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        plt.title("Comparaison des Courbes d'Apprentissage", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Épisodes", fontsize=13)
        plt.ylabel("Reward Cumulé Moyen", fontsize=13)
        plt.legend(fontsize=10, loc='lower right', framealpha=0.95)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('comparison_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Sauvegardé: comparison_learning_curves.png")
    
    def _plot_hyperparameter_sensitivity(self):
        """Graphique 2: Sensibilité aux hyperparamètres"""
        
        # Grouper par hyperparamètre varié
        hyperparam_groups = {}
        
        for result in self.results:
            for param in result.config.hyperparams:
                if param not in hyperparam_groups:
                    hyperparam_groups[param] = []
                hyperparam_groups[param].append(result)
        
        # Créer un subplot pour chaque hyperparamètre
        n_params = len(hyperparam_groups)
        if n_params == 0:
            print("⚠️ Aucune variation d'hyperparamètre détectée")
            return
        
        fig, axes = plt.subplots(1, min(n_params, 3), figsize=(18, 6))
        if n_params == 1:
            axes = [axes]
        
        for idx, (param_name, results) in enumerate(list(hyperparam_groups.items())[:3]):
            ax = axes[idx]
            
            # Extraire valeurs et performances
            param_values = []
            mean_rewards = []
            std_rewards = []
            
            for result in results:
                param_values.append(result.config.hyperparams.get(param_name))
                mean_rewards.append(result.final_metrics['mean_reward'])
                std_rewards.append(result.final_metrics['std_reward'])
            
            # Trier par valeur du paramètre
            sorted_indices = np.argsort(param_values)
            param_values = np.array(param_values)[sorted_indices]
            mean_rewards = np.array(mean_rewards)[sorted_indices]
            std_rewards = np.array(std_rewards)[sorted_indices]
            
            # Tracer
            ax.errorbar(param_values, mean_rewards, yerr=std_rewards, 
                       fmt='o-', capsize=5, linewidth=2.5, markersize=8,
                       color='#3498db', alpha=0.8)
            
            ax.set_title(f'Impact de {param_name}', fontsize=13, fontweight='bold')
            ax.set_xlabel(param_name, fontsize=11)
            ax.set_ylabel('Reward Final Moyen', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        fig.suptitle('Sensibilité aux Hyperparamètres', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('comparison_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Sauvegardé: comparison_sensitivity.png")
    
    def _plot_performance_comparison(self):
        """Graphique 3: Comparaison barres groupées"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['mean_reward', 'mean_success_rate', 'mean_latency']
        titles = ['Reward Final', 'Taux de Succès (%)', 'Latence Moyenne (ms)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            
            names = [r.config.name for r in self.results]
            values = [r.final_metrics[metric] for r in self.results]
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
            bars = ax.bar(range(len(names)), values, color=colors, 
                         alpha=0.8, edgecolor='black', linewidth=1.2)
            
            # Ajouter valeurs sur barres
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        fig.suptitle('Comparaison des Performances Finales', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('comparison_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Sauvegardé: comparison_performance.png")
    
    def _plot_stability_analysis(self):
        """Graphique 4: Analyse de stabilité (boxplots)"""
        plt.figure(figsize=(14, 8))
        
        data = [r.episode_rewards[-500:] for r in self.results]
        labels = [r.config.name for r in self.results]
        
        bp = plt.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        medianprops=dict(linewidth=2, color='red'))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title('Analyse de Stabilité (500 derniers épisodes)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Reward Cumulé', fontsize=13)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('comparison_stability.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Sauvegardé: comparison_stability.png")
    
    def _generate_summary_table(self):
        """Génère un tableau récapitulatif"""
        
        # Créer DataFrame
        data = []
        for result in self.results:
            row = {
                'Expérience': result.config.name,
                'Algorithme': result.config.algorithm,
                'Reward Final': f"{result.final_metrics['mean_reward']:.2f} ± {result.final_metrics['std_reward']:.2f}",
                'Taux Succès (%)': f"{result.final_metrics['mean_success_rate']:.1f}",
                'Latence (ms)': f"{result.final_metrics['mean_latency']:.2f}",
                'Variance': f"{result.final_metrics['variance_reward']:.2f}",
                'Temps (s)': f"{result.training_time:.1f}"
            }
            # Ajouter hyperparamètres
            for param, value in result.config.hyperparams.items():
                row[param] = value
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Afficher
        print("\n" + "="*120)
        print("📋 TABLEAU RÉCAPITULATIF")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120 + "\n")
        
        # Sauvegarder CSV
        df.to_csv('comparison_summary.csv', index=False)
        print("✓ Sauvegardé: comparison_summary.csv")
        
        # Sauvegarder JSON détaillé
        json_data = {
            'experiments': [r.config.to_dict() for r in self.results],
            'results': [{
                'config_name': r.config.name,
                'final_metrics': r.final_metrics,
                'training_time': r.training_time
            } for r in self.results]
        }
        
        with open('comparison_results.json', 'w') as f:
            json.dump(json_data, f, indent=2)
        print("✓ Sauvegardé: comparison_results.json")


# ============================================================================
# EXEMPLES D'UTILISATION
# ============================================================================

def example_compare_learning_rates():
    """
    Exemple: Comparer différents learning rates pour Double DQN
    """
    print("\n" + "="*80)
    print("📚 EXEMPLE: Comparaison Learning Rates - Double DQN")
    print("="*80)
    
    runner = ExperimentRunner()
    
    # Définir configurations avec différents learning rates
    learning_rates = [0.0001, 0.0003, 0.0005, 0.001]
    
    for lr in learning_rates:
        config = ExperimentConfig(
            name=f"DQN_lr_{lr}",
            algorithm="double_dqn",
            hyperparams={
                'learning_rate': lr,
                'epsilon_decay': 0.999,
                'target_update_freq': 500
            },
            n_episodes=2000,  # Réduit pour l'exemple
            description=f"Double DQN avec learning_rate={lr}"
        )
        
        runner.run_experiment(config)
    
    # Comparer
    runner.compare_results()


def example_compare_epsilon_decay():
    """
    Exemple: Comparer différentes stratégies d'exploration pour Q-Learning
    """
    print("\n" + "="*80)
    print("📚 EXEMPLE: Comparaison Epsilon Decay - Q-Learning")
    print("="*80)
    
    runner = ExperimentRunner()
    
    epsilon_decays = [0.995, 0.9975, 0.999, 0.9995]
    
    for decay in epsilon_decays:
        config = ExperimentConfig(
            name=f"QLearning_epsilon_{decay}",
            algorithm="qlearning",
            hyperparams={
                'epsilon_decay': decay,
                'lr': 0.05,
                'gamma': 0.99
            },
            n_episodes=2000,
            description=f"Q-Learning avec epsilon_decay={decay}"
        )
        
        runner.run_experiment(config)
    
    runner.compare_results()


def example_compare_ppo_configs():
    """
    Exemple: Comparer différentes configurations PPO
    """
    print("\n" + "="*80)
    print("📚 EXEMPLE: Comparaison Configurations - PPO")
    print("="*80)
    
    runner = ExperimentRunner()
    
    # Configuration 1: Standard
    config1 = ExperimentConfig(
        name="PPO_standard",
        algorithm="ppo",
        hyperparams={
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'batch_size': 128,
            'n_steps': 2048
        },
        total_timesteps=500000,
        description="Configuration standard PPO"
    )
    
    # Configuration 2: Learning rate élevé
    config2 = ExperimentConfig(
        name="PPO_high_lr",
        algorithm="ppo",
        hyperparams={
            'learning_rate': 0.001,
            'gamma': 0.99,
            'batch_size': 128,
            'n_steps': 2048
        },
        total_timesteps=500000,
        description="PPO avec learning rate élevé"
    )
    
    # Configuration 3: Plus de steps
    config3 = ExperimentConfig(
        name="PPO_more_steps",
        algorithm="ppo",
        hyperparams={
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'batch_size': 128,
            'n_steps': 4096
        },
        total_timesteps=500000,
        description="PPO avec plus de steps"
    )
    
    for config in [config1, config2, config3]:
        runner.run_experiment(config)
    
    runner.compare_results()


# ============================================================================
# MAIN - MENU INTERACTIF
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔬 SYSTÈME DE COMPARAISON D'HYPERPARAMÈTRES")
    print("="*80)
    print("""
    Ce système permet de:
    - Définir facilement des expériences avec différents hyperparamètres
    - Exécuter automatiquement plusieurs configurations
    - Comparer visuellement les résultats
    - Analyser l'impact de chaque hyperparamètre
    
    Exemples disponibles:
    1. Comparaison Learning Rates (Double DQN)
    2. Comparaison Epsilon Decay (Q-Learning)
    3. Comparaison Configurations (PPO)
    """)
    
    print("\nChoisissez un exemple à exécuter:")
    print("1. Comparaison Learning Rates - Double DQN")
    print("2. Comparaison Epsilon Decay - Q-Learning")
    print("3. Comparaison Configurations - PPO")
    print("4. Quitter")
    
    choice = input("\nVotre choix (1-4): ")
    
    if choice == "1":
        example_compare_learning_rates()
    elif choice == "2":
        example_compare_epsilon_decay()
    elif choice == "3":
        example_compare_ppo_configs()
    elif choice == "4":
        print("Au revoir!")
    else:
        print("⚠️ Choix invalide")
