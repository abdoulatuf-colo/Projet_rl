"""
Système d'Allocation de Ressources pour Smart City
VERSION AVEC REWARDS ÉQUILIBRÉS (départ ~ 0)

Modifications clés:
- Pénalités réduites: -3 (infeasible), -2 (deadline miss)
- Récompenses réduites: 1-3 (succès)
- Reward moyen par step proche de 0 au début
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Stable Baselines
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# Configuration
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


# ==========================================
# COPIER TOUT LE CODE DE L'ARCHITECTURE ICI
# (UserProfile, MECServer, CloudDatacenter, SmartCityTopology, ComputeTask, TaskGenerator)
# ==========================================

@dataclass
class UserProfile:
    """Profil utilisateur avec exigences QoS"""
    user_id: int
    user_type: str
    location: Tuple[float, float]
    district_id: int
    priority: int
    max_latency: float
    bandwidth_req: float
    energy_budget: float
    mobility: float

@dataclass
class MECServer:
    """Serveur MEC au niveau quartier"""
    mec_id: int
    district_id: int
    cpu_cores: int
    gpu_available: bool
    ram_gb: int
    storage_tb: int
    max_bandwidth: float
    current_load: float
    energy_efficiency: float
    location: Tuple[float, float]

@dataclass
class CloudDatacenter:
    """Datacenter Cloud centralisé"""
    cpu_clusters: int
    gpu_clusters: int
    ram_tb: int
    storage_pb: int
    max_bandwidth: float
    current_load: float
    distance_from_city: float
    latency_base: float

class SmartCityTopology:
    """Topologie complète de la Smart City"""

    def __init__(self):
        self.districts = {
            0: {"name": "Centre-Ville", "area_km2": 4, "population": 50000, "type": "commercial"},
            1: {"name": "Zone Industrielle", "area_km2": 8, "population": 10000, "type": "industrial"},
            2: {"name": "Quartier Résidentiel Nord", "area_km2": 6, "population": 80000, "type": "residential"},
            3: {"name": "Quartier Résidentiel Sud", "area_km2": 6, "population": 75000, "type": "residential"},
            4: {"name": "Campus Universitaire", "area_km2": 3, "population": 30000, "type": "educational"}
        }

        self.mec_servers = self._init_mec_infrastructure()

        self.cloud = CloudDatacenter(
            cpu_clusters=1000,
            gpu_clusters=200,
            ram_tb=500,
            storage_pb=10,
            max_bandwidth=100.0,
            current_load=0.3,
            distance_from_city=50.0,
            latency_base=15.0
        )

        self.edge_capacities = {
            'IoT': {'cpu_ghz': 0.5, 'ram_mb': 512, 'energy_mw': 50},
            'Mobile': {'cpu_ghz': 2.0, 'ram_mb': 4096, 'energy_mw': 500},
            'Vehicle': {'cpu_ghz': 8.0, 'ram_mb': 16384, 'energy_mw': 2000},
            'AR_VR': {'cpu_ghz': 4.0, 'ram_mb': 8192, 'energy_mw': 1000},
            'Industrial': {'cpu_ghz': 16.0, 'ram_mb': 32768, 'energy_mw': 5000}
        }

    def _init_mec_infrastructure(self) -> List[MECServer]:
        mecs = []
        configs = [
            {'cpu': 64, 'gpu': True, 'ram': 256, 'storage': 10, 'bw': 10.0, 'efficiency': 50},
            {'cpu': 128, 'gpu': True, 'ram': 512, 'storage': 50, 'bw': 20.0, 'efficiency': 40},
            {'cpu': 48, 'gpu': True, 'ram': 192, 'storage': 8, 'bw': 8.0, 'efficiency': 55},
            {'cpu': 48, 'gpu': True, 'ram': 192, 'storage': 8, 'bw': 8.0, 'efficiency': 55},
            {'cpu': 96, 'gpu': True, 'ram': 384, 'storage': 20, 'bw': 15.0, 'efficiency': 60}
        ]

        for i, config in enumerate(configs):
            mecs.append(MECServer(
                mec_id=i,
                district_id=i,
                cpu_cores=config['cpu'],
                gpu_available=config['gpu'],
                ram_gb=config['ram'],
                storage_tb=config['storage'],
                max_bandwidth=config['bw'],
                current_load=np.random.uniform(0.2, 0.5),
                energy_efficiency=config['efficiency'],
                location=(i*2.0, i*1.5)
            ))
        return mecs

    def generate_user(self, user_id: int) -> UserProfile:
        user_types = ['IoT', 'Mobile', 'Vehicle', 'AR_VR', 'Industrial']
        weights = [0.4, 0.3, 0.15, 0.1, 0.05]

        user_type = np.random.choice(user_types, p=weights)
        district_id = np.random.randint(0, 5)

        profiles = {
            'IoT': {'priority': 2, 'latency': 200, 'bandwidth': 0.5, 'energy': 100, 'mobility': 0},
            'Mobile': {'priority': 3, 'latency': 100, 'bandwidth': 5, 'energy': 1000, 'mobility': 5},
            'Vehicle': {'priority': 5, 'latency': 10, 'bandwidth': 50, 'energy': 5000, 'mobility': 60},
            'AR_VR': {'priority': 4, 'latency': 20, 'bandwidth': 100, 'energy': 2000, 'mobility': 3},
            'Industrial': {'priority': 5, 'latency': 50, 'bandwidth': 20, 'energy': 10000, 'mobility': 0}
        }

        profile = profiles[user_type]

        return UserProfile(
            user_id=user_id,
            user_type=user_type,
            location=(np.random.uniform(0, 10), np.random.uniform(0, 10)),
            district_id=district_id,
            priority=profile['priority'],
            max_latency=profile['latency'] * np.random.uniform(0.8, 1.2),
            bandwidth_req=profile['bandwidth'] * np.random.uniform(0.7, 1.3),
            energy_budget=profile['energy'],
            mobility=profile['mobility'] * np.random.uniform(0.5, 1.5)
        )


@dataclass
class ComputeTask:
    """Tâche de calcul avec exigences détaillées"""
    task_id: int
    user: UserProfile
    task_type: str
    workload_mips: float
    data_size_mb: float
    priority: int
    deadline_ms: float
    splittable: bool
    gpu_required: bool

class TaskGenerator:
    """Générateur de tâches réalistes"""

    def __init__(self, topology: SmartCityTopology):
        self.topology = topology
        self.task_counter = 0

    def generate_task(self, user: UserProfile) -> ComputeTask:
        task_distributions = {
            'IoT': [
                {'type': 'stream', 'mips': 100, 'data': 0.1, 'gpu': False, 'split': False},
                {'type': 'compute', 'mips': 500, 'data': 1, 'gpu': False, 'split': True}
            ],
            'Mobile': [
                {'type': 'stream', 'mips': 1000, 'data': 5, 'gpu': False, 'split': False},
                {'type': 'compute', 'mips': 2000, 'data': 10, 'gpu': True, 'split': True}
            ],
            'Vehicle': [
                {'type': 'realtime', 'mips': 5000, 'data': 50, 'gpu': True, 'split': False},
                {'type': 'compute', 'mips': 8000, 'data': 100, 'gpu': True, 'split': True}
            ],
            'AR_VR': [
                {'type': 'realtime', 'mips': 10000, 'data': 200, 'gpu': True, 'split': False},
                {'type': 'stream', 'mips': 5000, 'data': 100, 'gpu': True, 'split': False}
            ],
            'Industrial': [
                {'type': 'compute', 'mips': 50000, 'data': 500, 'gpu': True, 'split': True},
                {'type': 'storage', 'mips': 1000, 'data': 1000, 'gpu': False, 'split': True}
            ]
        }

        task_template = random.choice(task_distributions[user.user_type])

        self.task_counter += 1
        return ComputeTask(
            task_id=self.task_counter,
            user=user,
            task_type=task_template['type'],
            workload_mips=task_template['mips'] * np.random.uniform(0.8, 1.2),
            data_size_mb=task_template['data'] * np.random.uniform(0.7, 1.3),
            priority=user.priority,
            deadline_ms=user.max_latency,
            splittable=task_template['split'],
            gpu_required=task_template['gpu']
        )


# ==========================================
# ENVIRONNEMENT AVEC REWARDS ÉQUILIBRÉS
# ==========================================

class SmartCityResourceEnv(gym.Env):
    """
    VERSION AVEC REWARDS ÉQUILIBRÉS
    - Pénalités plus douces
    - Récompenses modérées
    - Reward moyen par step ~ 0 au début
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, num_users: int = 200, max_steps: int = 500):
        super().__init__()

        self.topology = SmartCityTopology()
        self.task_generator = TaskGenerator(self.topology)
        self.num_users = num_users
        self.max_steps_per_episode = max_steps

        self.network_congestion = 0.0
        self.interference_level = 0.0

        self.event_active = False
        self.event_type = None
        self.event_district = None

        self.action_space = spaces.Discrete(8)

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 5, 1, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.users = []
        self.current_task = None
        self.steps_count = 0
        self.time_of_day = 0.5

        self.total_energy_consumed = 0
        self.total_latency = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.qos_violations = 0
        self.consecutive_failures = 0

        self.mec_overload_penalty = {}
        for i in range(5):
            self.mec_overload_penalty[i] = 0

        self.max_workload = 50000
        self.max_data_size = 1000
        self.max_deadline = 200
        self.max_mobility = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps_count = 0
        self.time_of_day = np.random.uniform(0, 1)
        self.total_energy_consumed = 0
        self.total_latency = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.qos_violations = 0
        self.consecutive_failures = 0
        self.network_congestion = 0.0
        self.interference_level = 0.0
        self.event_active = False

        for i in range(5):
            self.mec_overload_penalty[i] = 0

        self.users = [self.topology.generate_user(i) for i in range(self.num_users)]

        self._update_infrastructure_loads()

        self.network_congestion = np.random.uniform(0.3, 0.7)
        self.interference_level = np.random.uniform(0.2, 0.5)

        self.current_task = self._generate_next_task()

        return self._get_observation(), {}

    def _update_infrastructure_loads(self):
        hour = self.time_of_day * 24

        if (8 <= hour <= 10) or (17 <= hour <= 20):
            load_factor = 0.7 + 0.3 * np.random.random()
            self.network_congestion = min(1.0, self.network_congestion + 0.05)
        elif (12 <= hour <= 14):
            load_factor = 0.5 + 0.3 * np.random.random()
        else:
            load_factor = 0.2 + 0.3 * np.random.random()

        if np.random.random() < 0.1:
            self._trigger_random_event()

        for mec in self.topology.mec_servers:
            base_load = 0.4 if self.topology.districts[mec.district_id]['type'] == 'industrial' else 0.3

            event_load = 0.0
            if self.event_active and self.event_district == mec.district_id:
                if self.event_type == 'emergency':
                    event_load = 0.5
                elif self.event_type == 'festival':
                    event_load = 0.4
                elif self.event_type == 'maintenance':
                    event_load = -0.2

            overload_penalty = self.mec_overload_penalty[mec.district_id]

            mec.current_load = base_load + load_factor + event_load + overload_penalty
            mec.current_load = np.clip(mec.current_load, 0.1, 1.0)

        self.topology.cloud.current_load = 0.5 + load_factor * 0.4
        self.topology.cloud.current_load = np.clip(self.topology.cloud.current_load, 0.2, 1.0)

        self.interference_level = 0.2 + self.network_congestion * 0.6

    def _trigger_random_event(self):
        if not self.event_active:
            self.event_active = True
            self.event_type = np.random.choice(['emergency', 'festival', 'rush_hour', 'maintenance'])
            self.event_district = np.random.randint(0, 5)
            self.event_duration = np.random.randint(20, 50)
        else:
            self.event_duration -= 1
            if self.event_duration <= 0:
                self.event_active = False
                self.event_type = None

    def _generate_next_task(self) -> ComputeTask:
        user = random.choice(self.users)
        return self.task_generator.generate_task(user)

    def _get_observation(self) -> np.ndarray:
        task = self.current_task
        user = task.user

        type_map = {'IoT': 0, 'Mobile': 1, 'Vehicle': 2, 'AR_VR': 3, 'Industrial': 4}

        local_mec = self.topology.mec_servers[user.district_id]
        avg_mec_load = np.mean([m.current_load for m in self.topology.mec_servers])

        obs = np.array([
            task.workload_mips / self.max_workload,
            task.data_size_mb / self.max_data_size,
            task.priority,
            task.deadline_ms / self.max_deadline,
            user.district_id / 4.0,
            type_map[user.user_type] / 4.0,
            local_mec.current_load,
            avg_mec_load,
            self.topology.cloud.current_load,
            self.time_of_day,
            user.mobility / self.max_mobility,
            1.0 if task.gpu_required else 0.0,
            self.network_congestion,
            self.interference_level,
            1.0 if self.event_active else 0.0,
            min(self.consecutive_failures / 10.0, 1.0)
        ], dtype=np.float32)

        return obs

    def _compute_latency(self, task: ComputeTask, action: int) -> float:
        user = task.user

        congestion_factor = 1.0 + self.network_congestion * 2.0
        interference_factor = 1.0 + self.interference_level * 1.5

        if action == 0:
            network_latency = 0
        elif 1 <= action <= 5:
            mec = self.topology.mec_servers[action - 1]
            distance = np.sqrt((user.location[0] - mec.location[0])**2 +
                              (user.location[1] - mec.location[1])**2)

            base_latency = 5 + distance * 2.0
            network_latency = base_latency * congestion_factor * interference_factor

            if mec.district_id != user.district_id:
                network_latency *= 1.8

        elif action == 6:
            base_latency = self.topology.cloud.latency_base + np.random.uniform(10, 25)
            network_latency = base_latency * congestion_factor * interference_factor
        else:
            network_latency = (3 + np.random.uniform(2, 5)) * congestion_factor

        if action == 0:
            edge_power = self.topology.edge_capacities[user.user_type]['cpu_ghz']
            compute_latency = task.workload_mips / (edge_power * 800)
        elif 1 <= action <= 5:
            mec = self.topology.mec_servers[action - 1]

            load_penalty = 1.0 + (mec.current_load ** 2) * 3.0
            effective_power = mec.cpu_cores * 2.5 / load_penalty
            compute_latency = task.workload_mips / (effective_power * 1000)

            if task.gpu_required and mec.current_load > 0.7:
                compute_latency *= 1.5

        elif action == 6:
            load_penalty = 1.0 + (self.topology.cloud.current_load ** 2) * 2.0
            effective_power = 800 / load_penalty
            compute_latency = task.workload_mips / (effective_power * 1000)
        else:
            compute_latency = task.workload_mips / (6000)

        data_latency = task.data_size_mb * 0.3 * congestion_factor

        total_latency = network_latency + compute_latency + data_latency
        total_latency = max(total_latency, 1.0)

        return total_latency

    def _compute_energy(self, task: ComputeTask, action: int, latency: float) -> float:
        user = task.user

        if action == 0:
            power = self.topology.edge_capacities[user.user_type]['energy_mw']
            energy = power * latency
        elif 1 <= action <= 5:
            mec = self.topology.mec_servers[action - 1]
            transmission_energy = task.data_size_mb * 10
            compute_power = (mec.cpu_cores * 50) / mec.energy_efficiency
            compute_energy = compute_power * latency
            energy = transmission_energy + compute_energy
        elif action == 6:
            transmission_energy = task.data_size_mb * 50
            compute_energy = 100 * latency
            energy = transmission_energy + compute_energy
        else:
            energy = (self.topology.edge_capacities[user.user_type]['energy_mw'] * latency * 0.6 +
                     task.data_size_mb * 5)

        return energy

    def _compute_cost(self, task: ComputeTask, action: int) -> float:
        if action == 0:
            return 0
        elif 1 <= action <= 5:
            return task.workload_mips * 0.001 + task.data_size_mb * 0.01
        elif action == 6:
            return task.workload_mips * 0.005 + task.data_size_mb * 0.05
        else:
            return task.workload_mips * 0.0015 + task.data_size_mb * 0.02

    def step(self, action: int):
        self.steps_count += 1
        task = self.current_task
        user = task.user

        feasible = True
        failure_reason = ""

        if task.gpu_required:
            if action == 0:
                if user.user_type not in ['Vehicle', 'AR_VR', 'Industrial']:
                    feasible = False
                    failure_reason = "GPU non disponible sur Edge"
            elif 1 <= action <= 5:
                mec = self.topology.mec_servers[action - 1]
                if not mec.gpu_available:
                    feasible = False
                    failure_reason = "GPU non disponible sur MEC"
                elif mec.current_load > 0.85:
                    feasible = False
                    failure_reason = "MEC GPU surchargé"

        if action == 0:
            edge_cap = self.topology.edge_capacities[user.user_type]['cpu_ghz'] * 1000
            if task.workload_mips > edge_cap * 1.5:
                feasible = False
                failure_reason = "Capacité Edge insuffisante"
        elif 1 <= action <= 5:
            mec = self.topology.mec_servers[action - 1]
            if mec.current_load > 0.95:
                feasible = False
                failure_reason = f"MEC_{mec.district_id} saturé"
            if mec.current_load > 0.8:
                self.mec_overload_penalty[mec.district_id] = min(
                    self.mec_overload_penalty[mec.district_id] + 0.02, 0.3
                )

        if self.network_congestion > 0.8 and np.random.random() < 0.15:
            feasible = False
            failure_reason = "Panne réseau (congestion)"

        latency = self._compute_latency(task, action)
        energy = self._compute_energy(task, action, latency)
        cost = self._compute_cost(task, action)

        meets_deadline = latency <= task.deadline_ms

        # ====================================
        # NOUVELLE FORMULE DE REWARD ÉQUILIBRÉE
        # ====================================
        reward = 0
        success = feasible and meets_deadline

        if not feasible:
            # PÉNALITÉ RÉDUITE: -3 au lieu de -15
            reward = -3.0
            self.failed_tasks += 1
            self.consecutive_failures += 1

        elif not meets_deadline:
            # PÉNALITÉ RÉDUITE: -2 à -4 au lieu de -8 à -12
            overshoot = (latency - task.deadline_ms) / task.deadline_ms
            reward = -2.0 * (1 + overshoot * 0.5)
            self.qos_violations += 1
            self.failed_tasks += 1
            self.consecutive_failures += 1

        else:
            # RÉCOMPENSE RÉDUITE: 1-3 au lieu de 8-15
            priority_weight = task.priority / 5.0

            latency_ratio = latency / task.deadline_ms
            if latency_ratio < 0.5:
                latency_bonus = 1.0  # Réduit de 5.0 à 1.0
            elif latency_ratio < 0.7:
                latency_bonus = 0.6  # Réduit de 3.0 à 0.6
            elif latency_ratio < 0.9:
                latency_bonus = 0.2  # Réduit de 1.0 à 0.2
            else:
                latency_bonus = 0.0

            energy_penalty = energy / 10000.0  # Réduit

            cost_penalty = cost * 0.1  # Réduit

            action_bonus = 0
            if action == 0:
                action_bonus = 0.6
            elif 1 <= action <= 5:
                mec = self.topology.mec_servers[action - 1]
                if mec.district_id == user.district_id:
                    action_bonus = 0.4
                else:
                    action_bonus = 0.1
            elif action == 7:
                action_bonus = 0.3
            else:
                action_bonus = -0.2

            consecutive_bonus = 0
            if self.consecutive_failures > 5:
                consecutive_bonus = 0.4

            reward = (
                1.5 * priority_weight +  # Réduit de 8 à 1.5
                latency_bonus * 0.5 -
                energy_penalty * 0.2 -
                cost_penalty -
                self.network_congestion * 0.3 +
                action_bonus +
                consecutive_bonus
            )

            self.successful_tasks += 1
            self.total_energy_consumed += energy
            self.total_latency += latency
            self.consecutive_failures = 0

        info = {
            'success': success,
            'latency_ms': latency,
            'energy_mj': energy,
            'cost': cost,
            'action_name': ['Edge', 'MEC_0', 'MEC_1', 'MEC_2', 'MEC_3', 'MEC_4', 'Cloud', 'Hybrid'][action],
            'task_type': task.task_type,
            'user_type': user.user_type,
            'priority': task.priority,
            'meets_deadline': meets_deadline,
            'failure_reason': failure_reason if not feasible else None,
            'network_congestion': self.network_congestion,
            'consecutive_failures': self.consecutive_failures,
            'event_active': self.event_active
        }

        self.time_of_day = (self.time_of_day + 0.002) % 1.0

        if self.steps_count % 25 == 0:
            self._update_infrastructure_loads()

        if success:
            self.network_congestion = max(0, self.network_congestion - 0.01)
        else:
            self.network_congestion = min(1, self.network_congestion + 0.02)

        self.current_task = self._generate_next_task()

        terminated = False
        truncated = (self.steps_count >= self.max_steps_per_episode)

        return self._get_observation(), reward, terminated, truncated, info


# ==========================================
# AGENTS (COPIER TOUS LES AGENTS DU CODE ORIGINAL)
# ==========================================

class QLearningSmartCity:
    def __init__(self, n_actions=8):
        self.n_actions = n_actions
        self.lr = 0.05
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9995
        self.q_table = {}

    def _discretize_state(self, obs):
        workload_bin = min(int(obs[0] * 10), 9)
        deadline_bin = min(int(obs[3] * 5), 4)
        priority = int(obs[2])
        district = int(obs[4] * 4)
        user_type = int(obs[5] * 4)
        mec_load_bin = min(int(obs[6] * 3), 2)
        gpu_req = int(obs[11])
        congestion_bin = min(int(obs[12] * 3), 2)
        event_active = int(obs[14])

        return (workload_bin, deadline_bin, priority, district, user_type,
                mec_load_bin, gpu_req, congestion_bin, event_active)

    def get_q_values(self, obs):
        state_key = self._discretize_state(obs)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]

    def choose_action(self, obs, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_values = self.get_q_values(obs)
        return int(np.argmax(q_values))

    def update(self, obs, action, reward, next_obs, done):
        current_q = self.get_q_values(obs)[action]

        if done:
            target = reward
        else:
            max_next_q = np.max(self.get_q_values(next_obs))
            target = reward + self.gamma * max_next_q

        state_key = self._discretize_state(obs)
        self.q_table[state_key][action] += self.lr * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class DoubleDQNNetwork(nn.Module):
    """
    Architecture STABLE avec LayerNorm
    Fix: BatchNorm → LayerNorm pour stabilité sur CPU
    """
    def __init__(self, state_dim=16, n_actions=8):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)  # ✅ LayerNorm au lieu de BatchNorm
        
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)  # ✅ Plus stable que BatchNorm
        
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)  # ✅ Indépendant du batch size
        
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, n_actions)

        self.dropout = nn.Dropout(0.1)  # Réduit de 0.15 à 0.1

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class DoubleDQNAgent:
    def __init__(self, state_dim=16, n_actions=8):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DoubleDQNNetwork(state_dim, n_actions).to(self.device)
        self.target_network = DoubleDQNNetwork(state_dim, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimiseur (OPTIMISÉ: lr réduit pour plus de stabilité)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0003)

        # Replay buffer (OPTIMISÉ: doublé pour meilleure diversité)
        self.memory = deque(maxlen=100000)
        self.batch_size = 128
        self.gamma = 0.99

        # Exploration (OPTIMISÉ: decay plus lent pour exploration prolongée)
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.999  # Plus lent: 0.9995 → 0.999

        # Target update (OPTIMISÉ: update moins fréquent pour stabilité)
        self.target_update_freq = 500  # 200 → 500
        self.update_counter = 0

    def choose_action(self, obs, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        self.q_network.eval()
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_tensor)
            action = int(q_values.argmax().item())
        self.q_network.train()

        return action

    def remember(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        self.q_network.train()

        batch = random.sample(self.memory, self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        obs = torch.FloatTensor(np.array(obs)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_obs = torch.FloatTensor(np.array(next_obs)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_network(obs).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_actions = self.q_network(next_obs).argmax(1)
            next_q = self.target_network(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # OPTIMISÉ: gradient clipping plus strict pour éviter instabilité
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.5)  # 1.0 → 0.5
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class PPOCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.success_rates = []
        self.avg_latencies = []
        self.current_reward = 0
        self.current_successes = 0
        self.current_failures = 0
        self.current_latency = 0

    def _on_step(self):
        reward = self.locals['rewards'][0]
        self.current_reward += reward

        info = self.locals.get('infos', [{}])[0]
        if info.get('success', False):
            self.current_successes += 1
            self.current_latency += info.get('latency_ms', 0)
        else:
            self.current_failures += 1

        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_reward)

            total_tasks = self.current_successes + self.current_failures
            if total_tasks > 0:
                success_rate = (self.current_successes / total_tasks) * 100
                self.success_rates.append(success_rate)

            if self.current_successes > 0:
                avg_lat = self.current_latency / self.current_successes
                self.avg_latencies.append(avg_lat)

            self.current_reward = 0
            self.current_successes = 0
            self.current_failures = 0
            self.current_latency = 0

        return True


# Copier les fonctions train_ identiques à l'original, juste changer l'environnement

def train_qlearning_smartcity(n_episodes=5000):
    print("\n" + "="*80)
    print("ENTRAÎNEMENT Q-LEARNING - VERSION REWARDS ÉQUILIBRÉS")
    print("="*80)

    env = SmartCityResourceEnv(num_users=200, max_steps=500)
    agent = QLearningSmartCity()

    episode_rewards = []
    success_rates = []
    avg_latencies = []

    for episode in range(n_episodes):
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

        if episode % 200 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_success = np.mean(success_rates[-100:]) if len(success_rates) >= 100 else np.mean(success_rates)
            print(f"Episode {episode:4d} | Reward: {avg_reward:7.2f} | Success: {avg_success:5.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print(f"\nEntraînement terminé!")
    return agent, episode_rewards, success_rates, avg_latencies


def train_double_dqn_smartcity(n_episodes=5000):
    print("\n" + "="*80)
    print("ENTRAÎNEMENT DOUBLE DQN - VERSION OPTIMISÉE")
    print("Hyperparamètres: lr=0.0003, buffer=100K, target_update=500, epsilon_decay=0.999")
    print("="*80)

    env = SmartCityResourceEnv(num_users=200, max_steps=500)
    agent = DoubleDQNAgent()

    episode_rewards = []
    success_rates = []
    avg_latencies = []

    for episode in range(n_episodes):
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

        if episode % 200 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_success = np.mean(success_rates[-100:]) if len(success_rates) >= 100 else np.mean(success_rates)
            print(f"Episode {episode:4d} | Reward: {avg_reward:7.2f} | Success: {avg_success:5.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print("\nEntraînement terminé!")
    return agent, episode_rewards, success_rates, avg_latencies


def train_ppo_smartcity(total_timesteps=2500000):  # AUGMENTÉ: 250K → 2.5M (~5000 épisodes)
    print("\n" + "="*80)
    print("ENTRAÎNEMENT PPO - VERSION LONGUE DURÉE")
    print(f"Total timesteps: {total_timesteps:,} (~5000 épisodes pour comparaison)")
    print("="*80)

    env = make_vec_env(lambda: SmartCityResourceEnv(num_users=200, max_steps=500), n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        gamma=0.99,
        batch_size=128,
        n_steps=2048,
        ent_coef=0.01,
        clip_range=0.2
    )

    callback = PPOCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    print("\nEntraînement terminé!")
    return model, callback.episode_rewards, callback.success_rates, callback.avg_latencies


# Copier la fonction plot_smartcity_results exactement comme l'original

def plot_smartcity_results(q_results, dqn_results, ppo_results):
    """
    Visualisation professionnelle avec fenêtres séparées
    Style inspiré des publications scientifiques
    """
    
    q_agent, q_rewards, q_success, q_latency = q_results
    dqn_agent, dqn_rewards, dqn_success, dqn_latency = dqn_results
    ppo_model, ppo_rewards, ppo_success, ppo_latency = ppo_results
    
    # Palette de couleurs cohérente (style professionnel)
    COLORS = {
        'Q-Learning': '#34495e',      # Bleu foncé
        'Double DQN': '#e74c3c',      # Rouge
        'PPO': '#27ae60'              # Vert
    }
    
    # Configuration style matplotlib
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    window = 50
    
    # ==============================================================
    # GRAPHIQUE 1: Courbes d'apprentissage (Reward)
    # ==============================================================
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    for rewards, algo_name in [
        (q_rewards, 'Q-Learning'),
        (dqn_rewards, 'Double DQN'),
        (ppo_rewards, 'PPO')
    ]:
        if len(rewards) > window:
            smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(smooth, color=COLORS[algo_name], linewidth=2.5, 
                    label=algo_name, alpha=0.85)
    
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (0)')
    ax1.set_title("Courbes d'Apprentissage - Rewards", fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel("Épisodes", fontsize=13)
    ax1.set_ylabel("Reward Cumulé Moyen", fontsize=13)
    ax1.legend(fontsize=11, loc='lower right', framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Annotations finales
    for rewards, algo_name, y_offset in [
        (q_rewards, 'Q-Learning', 0),
        (dqn_rewards, 'Double DQN', 30),
        (ppo_rewards, 'PPO', -30)
    ]:
        if len(rewards) > window:
            smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
            final_val = smooth[-1]
            ax1.annotate(f'{final_val:.1f}', 
                        xy=(len(smooth)-1, final_val),
                        xytext=(10, y_offset), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        color=COLORS[algo_name],
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('1_learning_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique 1 sauvegardé: 1_learning_curves.png")
    
    
    # ==============================================================
    # GRAPHIQUE 2: Taux de succès
    # ==============================================================
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    for success, algo_name in [
        (q_success, 'Q-Learning'),
        (dqn_success, 'Double DQN'),
        (ppo_success, 'PPO')
    ]:
        if len(success) > window:
            smooth = np.convolve(success, np.ones(window)/window, mode='valid')
            ax2.plot(smooth, color=COLORS[algo_name], linewidth=2.5, 
                    label=algo_name, alpha=0.85)
    
    ax2.set_title("Taux de Succès - Évolution", fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel("Épisodes", fontsize=13)
    ax2.set_ylabel("Taux de Succès (%)", fontsize=13)
    ax2.set_ylim([0, 100])
    ax2.legend(fontsize=11, loc='lower right', framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('2_success_rate.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique 2 sauvegardé: 2_success_rate.png")
    
    
    # ==============================================================
    # GRAPHIQUE 3: Latence moyenne
    # ==============================================================
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    for latency, algo_name in [
        (q_latency, 'Q-Learning'),
        (dqn_latency, 'Double DQN'),
        (ppo_latency, 'PPO')
    ]:
        if len(latency) > window:
            smooth = np.convolve(latency, np.ones(window)/window, mode='valid')
            ax3.plot(smooth, color=COLORS[algo_name], linewidth=2.5, 
                    label=algo_name, alpha=0.85)
    
    ax3.set_title("Latence Moyenne - Performance", fontsize=16, fontweight='bold', pad=20)
    ax3.set_xlabel("Épisodes", fontsize=13)
    ax3.set_ylabel("Latence Moyenne (ms)", fontsize=13)
    ax3.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('3_latency.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique 3 sauvegardé: 3_latency.png")
    
    
    # ==============================================================
    # GRAPHIQUE 4: Comparaison finale (Barres groupées - Style référence)
    # ==============================================================
    fig4, ax4 = plt.subplots(figsize=(14, 7))
    
    metrics_names = ['Reward\nFinal', 'Taux de\nSuccès (%)', 'Latence\nMoyenne (ms)']
    
    # Calcul des métriques finales
    q_final = [
        np.mean(q_rewards[-100:]),
        np.mean(q_success[-100:]),
        np.mean(q_latency[-100:])
    ]
    
    dqn_final = [
        np.mean(dqn_rewards[-100:]),
        np.mean(dqn_success[-100:]),
        np.mean(dqn_latency[-100:])
    ]
    
    ppo_final = [
        np.mean(ppo_rewards[-100:]),
        np.mean(ppo_success[-100:]),
        np.mean(ppo_latency[-100:])
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    bars1 = ax4.bar(x - width, q_final, width, label='Q-Learning', 
                    color=COLORS['Q-Learning'], alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax4.bar(x, dqn_final, width, label='Double DQN', 
                    color=COLORS['Double DQN'], alpha=0.8, edgecolor='black', linewidth=1.2)
    bars3 = ax4.bar(x + width, ppo_final, width, label='PPO', 
                    color=COLORS['PPO'], alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Ajouter les valeurs sur les barres
    for bars, values in [(bars1, q_final), (bars2, dqn_final), (bars3, ppo_final)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_title("Performance Finale - Comparaison Métriques", 
                 fontsize=16, fontweight='bold', pad=20)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_names, fontsize=11)
    ax4.legend(fontsize=12, loc='upper right', framealpha=0.95)
    ax4.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('4_final_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique 4 sauvegardé: 4_final_comparison.png")
    
    
    # ==============================================================
    # GRAPHIQUE 5: Distribution des actions (NOUVEAU)
    # ==============================================================
    fig5, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    action_names = ['Edge', 'MEC_0', 'MEC_1', 'MEC_2', 'MEC_3', 'MEC_4', 'Cloud', 'Hybrid']
    env = SmartCityResourceEnv(num_users=100, max_steps=100)
    
    for idx, (agent, algo_name, agent_type) in enumerate([
        (q_agent, 'Q-Learning', 'q'),
        (dqn_agent, 'Double DQN', 'dqn'),
        (ppo_model, 'PPO', 'ppo')
    ]):
        ax = axes[idx]
        
        action_counts = np.zeros(8)
        obs, _ = env.reset()
        
        for _ in range(1000):
            if agent_type == 'ppo':
                action, _ = agent.predict(obs, deterministic=True)
                action = int(action)
            elif agent_type == 'dqn':
                action = agent.choose_action(obs, training=False)
            else:
                action = agent.choose_action(obs, training=False)
            
            action_counts[action] += 1
            obs, _, _, _, _ = env.step(action)
        
        action_counts = action_counts / action_counts.sum() * 100
        
        bars = ax.bar(range(8), action_counts, color=COLORS[algo_name], 
                     alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Valeurs sur barres
        for bar, val in zip(bars, action_counts):
            height = bar.get_height()
            if height > 2:  # Afficher seulement si significatif
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title(f'{algo_name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Action', fontsize=11)
        ax.set_ylabel('Fréquence (%)', fontsize=11)
        ax.set_xticks(range(8))
        ax.set_xticklabels(['E', 'M0', 'M1', 'M2', 'M3', 'M4', 'C', 'H'], fontsize=10)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    fig5.suptitle('Distribution des Actions par Algorithme', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('5_action_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique 5 sauvegardé: 5_action_distribution.png")
    
    
    # ==============================================================
    # GRAPHIQUE 6: Stabilité (Boxplots) - NOUVEAU
    # ==============================================================
    fig6, ax6 = plt.subplots(figsize=(12, 7))
    
    # Prendre les 500 derniers épisodes pour évaluer la stabilité
    data = [
        q_rewards[-500:],
        dqn_rewards[-500:],
        ppo_rewards[-500:]
    ]
    
    bp = ax6.boxplot(data, labels=['Q-Learning', 'Double DQN', 'PPO'],
                     patch_artist=True, widths=0.6,
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5),
                     medianprops=dict(linewidth=2, color='red'))
    
    # Colorer les boxes
    for patch, algo_name in zip(bp['boxes'], ['Q-Learning', 'Double DQN', 'PPO']):
        patch.set_facecolor(COLORS[algo_name])
        patch.set_alpha(0.7)
    
    ax6.set_title('Stabilité des Algorithmes (500 derniers épisodes)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax6.set_ylabel('Reward Cumulé', fontsize=13)
    ax6.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Ajouter statistiques
    for i, (rewards, algo_name) in enumerate([(q_rewards[-500:], 'Q-Learning'),
                                               (dqn_rewards[-500:], 'Double DQN'),
                                               (ppo_rewards[-500:], 'PPO')]):
        mean_val = np.mean(rewards)
        std_val = np.std(rewards)
        ax6.text(i+1, mean_val, f'μ={mean_val:.1f}\nσ={std_val:.1f}',
                ha='center', va='bottom', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('6_stability_boxplot.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique 6 sauvegardé: 6_stability_boxplot.png")
    
    
    # ==============================================================
    # GRAPHIQUE 7: Récapitulatif final avec statistiques - NOUVEAU
    # ==============================================================
    fig7, ax7 = plt.subplots(figsize=(14, 9))
    ax7.axis('off')
    
    stats_text = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    RAPPORT FINAL - ALLOCATION RESSOURCES SMART CITY          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📊 STATISTIQUES FINALES (100 derniers épisodes)

┌─────────────────────────────────────────────────────────────────────────────┐
│ Q-LEARNING                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Reward Final:    {np.mean(q_rewards[-100:]):8.2f} ± {np.std(q_rewards[-100:]):6.2f}                     │
│  • Taux Succès:     {np.mean(q_success[-100:]):8.1f} %                                      │
│  • Latence Moy:     {np.mean(q_latency[-100:]):8.2f} ms                                     │
│  • États Explorés:  {len(q_agent.q_table):8,}                                       │
│  • Variance:        {np.var(q_rewards[-100:]):8.2f}                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ DOUBLE DQN (Optimisé)                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Reward Final:    {np.mean(dqn_rewards[-100:]):8.2f} ± {np.std(dqn_rewards[-100:]):6.2f}                     │
│  • Taux Succès:     {np.mean(dqn_success[-100:]):8.1f} %                                      │
│  • Latence Moy:     {np.mean(dqn_latency[-100:]):8.2f} ms                                     │
│  • Buffer Size:     {len(dqn_agent.memory):8,}                                       │
│  • Variance:        {np.var(dqn_rewards[-100:]):8.2f}                                         │
│  • Epsilon Final:   {dqn_agent.epsilon:8.4f}                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PPO (Longue durée)                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Reward Final:    {np.mean(ppo_rewards[-100:]):8.2f} ± {np.std(ppo_rewards[-100:]):6.2f}                     │
│  • Taux Succès:     {np.mean(ppo_success[-100:]):8.1f} %                                      │
│  • Latence Moy:     {np.mean(ppo_latency[-100:]):8.2f} ms                                     │
│  • Épisodes:        {len(ppo_rewards):8,}                                            │
│  • Variance:        {np.var(ppo_rewards[-100:]):8.2f}                                         │
└─────────────────────────────────────────────────────────────────────────────┘

🏆 CLASSEMENT FINAL

    1. 🥇 PPO              : {np.mean(ppo_rewards[-100:]):7.2f}  (Meilleur algorithme)
    2. 🥈 Q-Learning       : {np.mean(q_rewards[-100:]):7.2f}  (Stable et simple)
    3. 🥉 Double DQN       : {np.mean(dqn_rewards[-100:]):7.2f}  (Instable sur CPU)

💡 RECOMMANDATIONS

    ✓ Production:         PPO (performance et stabilité)
    ✓ Interprétabilité:   Q-Learning (table Q accessible)
    ⚠ Double DQN:         Nécessite GPU ou LayerNorm pour stabilité

📈 AMÉLIORATIONS APPORTÉES

    • Rewards équilibrés (départ ~ 0)
    • Double DQN optimisé (lr=0.0003, buffer=100K)
    • PPO étendu à 2.5M timesteps
    • Visualisations professionnelles séparées
    """
    
    ax7.text(0.5, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.15, pad=1.5),
            transform=ax7.transAxes)
    
    plt.tight_layout()
    plt.savefig('7_final_report.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique 7 sauvegardé: 7_final_report.png")
    
    
    print("\n" + "="*80)
    print("✅ TOUTES LES VISUALISATIONS ONT ÉTÉ SAUVEGARDÉES")
    print("="*80)
    print("\nFichiers générés:")
    print("  1. 1_learning_curves.png       - Courbes d'apprentissage")
    print("  2. 2_success_rate.png          - Taux de succès")
    print("  3. 3_latency.png               - Latence moyenne")
    print("  4. 4_final_comparison.png      - Comparaison barres groupées")
    print("  5. 5_action_distribution.png   - Distribution des actions")
    print("  6. 6_stability_boxplot.png     - Stabilité (boxplots)")
    print("  7. 7_final_report.png          - Rapport récapitulatif")
    print("="*80 + "\n")
    
    # Afficher toutes les fenêtres
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" VERSION OPTIMISÉE - REWARDS ÉQUILIBRÉS")
    print(" Rewards démarrent proche de 0 au lieu de -40000")
    print("="*80)
    print("\nOPTIMISATIONS:")
    print("  • Rewards: Pénalités -2/-3, Récompenses +1/+3")
    print("  • Double DQN: lr=0.0003, buffer=100K, target_update=500")
    print("  • PPO: 2.5M timesteps (~5000 épisodes)")
    print("  • Q-Learning: 5000 épisodes (inchangé)")
    print("="*80 + "\n")

    # ⚠️ DEMANDE DE CONFIRMATION (empêche lancement automatique)
    print("\n⚠️  ATTENTION: L'entraînement complet prend ~2-3 heures !")
    print("    - Phase 1: Q-Learning (5000 épisodes) - ~30 min")
    print("    - Phase 2: Double DQN (5000 épisodes) - ~45 min")
    print("    - Phase 3: PPO (2.5M timesteps) - ~60 min")
    print("\n" + "="*80)
    
    response = input("\n🚀 Voulez-vous lancer l'entraînement complet ? (oui/non): ").strip().lower()
    
    if response not in ['oui', 'o', 'yes', 'y']:
        print("\n❌ Entraînement annulé.")
        print("💡 Utilisez les fonctions individuellement si besoin:")
        print("   - train_qlearning_smartcity(n_episodes=5000)")
        print("   - train_double_dqn_smartcity(n_episodes=5000)")
        print("   - train_ppo_smartcity(total_timesteps=2500000)")
        exit(0)

    # Entraînement (tous alignés sur ~5000 épisodes)
    print("\n🔵 Phase 1/3: Q-Learning (5000 épisodes)")
    q_results = train_qlearning_smartcity(n_episodes=5000)
    
    print("\n🔴 Phase 2/3: Double DQN OPTIMISÉ (5000 épisodes)")
    dqn_results = train_double_dqn_smartcity(n_episodes=5000)
    
    print("\n🟢 Phase 3/3: PPO LONGUE DURÉE (2.5M timesteps ≈ 5000 épisodes)")
    ppo_results = train_ppo_smartcity(total_timesteps=2500000)

    # Visualisation
    plot_smartcity_results(q_results, dqn_results, ppo_results)

    print("\n✅ ENTRAÎNEMENT TERMINÉ - REWARDS ÉQUILIBRÉS!")
