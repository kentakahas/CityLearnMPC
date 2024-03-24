import os
import joblib
import numpy as np

# type hinting
from typing import Any

from preprocessing import get_datetime, set_schema_simulation_period
from models.electric_pricing import generate_24_price

from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv

# Set Random seed
np.random.seed(42)


class MPC(Agent):
    def __init__(self, schema: dict, env: CityLearnEnv,
                 model_folder: str,
                 lookback: int = 12, horizon: int = 24,
                 pop: int = 50,
                 **kwargs: Any):
        super().__init__(env, **kwargs)
        self.schema = schema
        self.env = env
        self.lookback = lookback
        self.horizon = horizon
        self.b_names = [b.name for b in self.env.buildings]
        self.current_timestep = self.env.simulation_start_time_step
        self.pop = pop

        # load prediction models
        self.model_folder = model_folder
        self.load_predictor = self._load_models("load_12.pkl")
        self.gen_predictor = self._load_models("gen_12.pkl")
        self.netelec_predictor = self._load_models("netelec.pkl")
        self.carbon_predictor = joblib.load(os.path.join(self.model_folder, "carbon.pkl"))

        # initialize dictionaries
        self.previous_states = {
            b_name: {
                'non_shiftable_load': np.zeros(self.lookback + 1),
                'solar_generation': np.zeros(self.lookback + 1)
            } for b_name in self.b_names
        }

        self.current_SoC = {
            b_name: b.observations()['electrical_storage_soc'] for b, b_name in zip(self.env.buildings, self.b_names)
        }

        self.future_states = {
            b_name: {
                'non_shiftable_load': np.zeros(self.horizon),
                'solar_generation': np.zeros(self.horizon)
            } for b_name in self.b_names
        }

        self.baseline_net_electricity = {b: np.zeros(self.horizon) for b in self.b_names}

        self._future_actions = {b_name: np.zeros((1, self.horizon)) for b_name in self.b_names}
        self.future_net_electricity = {b: np.zeros(self.horizon) for b in self.b_names}

        # update dictionaries for initial step
        self._warmup(self.lookback)
        self._predict_future_states()

    @property
    def low_bound(self) -> float:
        return self.env.action_space[0].low[0]

    @property
    def high_bound(self) -> float:
        return self.env.action_space[0].high[0]

    @property
    def future_actions(self) -> dict[str, np.ndarray]:
        return self._future_actions

    @future_actions.setter
    def future_actions(self, future_actions: dict):
        self._future_actions = future_actions

    def _load_models(self, model_name: str):
        models = {}
        for b_name in self.b_names:
            models[b_name] = joblib.load(os.path.join(
                self.model_folder, b_name, model_name))
        return models

    def _warmup(self, warmup_timesteps: int):
        warmup_schema = self.schema.copy()
        stop_time = self.schema['simulation_start_time_step']
        start_time = stop_time - warmup_timesteps
        warmup_schema = set_schema_simulation_period(warmup_schema, start_time, stop_time)
        warmup_env = CityLearnEnv(warmup_schema)
        for b in warmup_env.buildings:
            self.previous_states[b.name]["non_shiftable_load"][0] = b.observations()["non_shiftable_load"]
            self.previous_states[b.name]["solar_generation"][0] = b.observations()["solar_generation"]

        while not warmup_env.done:
            actions = [0.0] * len(self.b_names)
            warmup_env.step([actions])
            step = warmup_env.time_step
            for b in warmup_env.buildings:
                self.previous_states[b.name]["non_shiftable_load"][step] = b.observations()["non_shiftable_load"]
                self.previous_states[b.name]["solar_generation"][step] = b.observations()["solar_generation"]

    def _load_input(self, b_name):
        input_features = ['month', 'day_type', 'hour']
        input_features = np.array([self.env.buildings[0].observations()[k] for k in input_features])
        prev_state = self.previous_states[b_name]["non_shiftable_load"]
        return np.concatenate((input_features, prev_state), axis=0).reshape(1, -1)

    def _gen_input(self, b_name):
        input_features = ['month', 'day_type', 'hour',
                          "diffuse_solar_irradiance",
                          "diffuse_solar_irradiance_predicted_6h",
                          "diffuse_solar_irradiance_predicted_12h",
                          "direct_solar_irradiance",
                          "direct_solar_irradiance_predicted_6h",
                          "direct_solar_irradiance_predicted_12h"]
        input_features = np.array([self.env.buildings[0].observations()[k] for k in input_features])
        prev_state = self.previous_states[b_name]["solar_generation"]
        return np.concatenate((input_features, prev_state), axis=0).reshape(1, -1)

    def _predict_future_states(self):
        for b_name in self.b_names:
            self.future_states[b_name]["non_shiftable_load"] = \
                self.load_predictor[b_name].predict(self._load_input(b_name))[0]
            self.future_states[b_name]["solar_generation"] = \
                self.gen_predictor[b_name].predict(self._gen_input(b_name))[0]

            self.baseline_net_electricity[b_name] = \
                self.future_states[b_name]["non_shiftable_load"] - self.future_states[b_name]["solar_generation"]

    def _update_current_soc(self):
        for b in self.env.buildings:
            self.current_SoC[b.name] = b.observations()["electrical_storage_soc"]

    def _update_previous_states(self):
        for b in self.env.buildings:
            for key in ["non_shiftable_load", "solar_generation"]:
                previous_state = self.previous_states[b.name][key]
                observations = b.observations()[key]

                previous_state[:-1] = previous_state[1:]
                previous_state[-1] = observations

    def _update_future_actions(self):
        for b_name in self.b_names:
            new_action = np.zeros((1, 1))
            self.future_actions[b_name] = np.concatenate([self.future_actions[b_name][:, 1:], new_action], axis=1)

    def update_mpc(self):
        self._update_current_soc()
        self._update_previous_states()
        self._predict_future_states()
        self._update_future_actions()
        self.current_timestep += 1

    @property
    def future_electricity_pricing(self):
        return np.array(generate_24_price(get_datetime(self.current_timestep)))

    @property
    def future_carbon_intensity(self):
        input_features = ['month', 'day_type', 'hour', 'carbon_intensity']
        features = np.array([[self.env.buildings[0].observations()[k] for k in input_features]])
        return self.carbon_predictor.predict(features)[0]

    @property
    def baseline_total_net_electricity(self):
        return sum(self.baseline_net_electricity[b] for b in self.b_names)

    # Evaluation
    def predict_net_electricity(self, actions, future_states=None):
        if future_states is None:
            future_states = self.future_states
        results = {}
        soc_hist = {}

        for b_name in self.b_names:
            rows = actions[b_name].shape[0]
            netelecs = np.zeros((rows, self.horizon))
            socs = np.zeros((rows, self.horizon + 1))
            socs[:, 0] = self.current_SoC[b_name]
            b_loads = future_states[b_name]["non_shiftable_load"]
            b_gen = future_states[b_name]["solar_generation"]
            b_actions = actions[b_name]
            b_predictor = self.netelec_predictor[b_name]

            for i in range(self.horizon):
                loads = np.full(rows, b_loads[i])
                gens = np.full(rows, b_gen[i])
                acts = b_actions[:, i]
                inputs = np.column_stack([loads, gens, socs[:, i], acts])

                # Assuming self.netelec_predictor[b_name].predict is a vectorized function
                predicts = b_predictor.predict(inputs)

                socs[:, i + 1] = predicts[:, 0]
                netelecs[:, i] = predicts[:, 1]

            results[b_name] = netelecs
            soc_hist[b_name] = socs

        return results, soc_hist

    def total_net_electricity(self, netelecs):
        return sum(netelecs[b] for b in self.b_names)

    def energy_score(self, total_nets):
        control = np.sum(total_nets, axis=1)
        baseline = np.sum(self.baseline_total_net_electricity)
        return control / baseline

    def price_score(self, total_nets, future_electricity=None):
        if future_electricity is None:
            future_electricity = self.future_electricity_pricing
        control = np.dot(np.maximum(total_nets, 0), future_electricity)
        baseline = np.dot(np.maximum(self.baseline_total_net_electricity, 0), future_electricity)
        return control / baseline

    def carbon_score(self, total_nets, future_carbon=None):
        if future_carbon is None:
            future_carbon = self.future_carbon_intensity
        control = np.dot(np.maximum(total_nets, 0), future_carbon)
        baseline = np.dot(np.maximum(self.baseline_total_net_electricity, 0), future_carbon)
        return control / baseline

    def average_score(self, total_nets, future_electricity=None, future_carbon=None):
        # energy = self.energy_score(total_nets)
        price = self.price_score(total_nets, future_electricity)
        carbon = self.carbon_score(total_nets, future_carbon)
        return (price + carbon) / 2

    def penalty_function(self, actions: dict[str, np.ndarray], soc):
        penalty = 0
        for b_name in self.b_names:
            expected = actions[b_name] + soc[b_name][:, :24]
            penalty += np.sum(np.abs(expected - soc[b_name][:, 1:]), axis=1)
        return penalty

    def objective_function(self, actions: dict[str, np.ndarray],
                           future_states: dict = None,
                           future_electricity=None, future_carbon=None
                           ) -> np.ndarray:
        net, soc = self.predict_net_electricity(actions, future_states)
        total_net = self.total_net_electricity(net)
        scores = self.average_score(total_net, future_electricity, future_carbon)
        penalty = self.penalty_function(actions, soc)
        return scores + penalty

    # optimizer
    def _generate_population(self):
        population = {}
        for b_name in self.b_names:
            actions = []
            state = np.full(self.pop - 1, self.current_SoC[b_name])
            for _ in range(self.horizon):
                action = np.random.uniform(np.maximum(-1 * state, self.low_bound),
                                           np.minimum(1 - state, self.high_bound),
                                           size=(self.pop - 1))
                actions.append(action)
                state += action
            population[b_name] = np.vstack([self.future_actions[b_name], np.array(actions).T])
        return population

    def _sample_pop(self, size):
        population_index = np.arange(self.pop)
        choices = np.tile(population_index, (self.pop, 1))
        np.fill_diagonal(choices, -1)

        def choose_row(row):
            return np.random.choice(row[row != -1], size=size, replace=False)

        indices = np.apply_along_axis(choose_row, axis=1, arr=choices)

        return indices

    def _mutate(self, actions, best, strategy: str, f: float | tuple):
        if type(f) is tuple:
            scaler = np.random.uniform(f[0], f[1], size=(actions.shape[0], 1))
        else:
            scaler = f

        def _rand1(r):
            indices = self._sample_pop(3)
            x = r[indices.T]
            return x[0] + scaler * (x[1] + x[2])

        def _rand2(r):
            indices = self._sample_pop(5)
            x = r[indices.T]
            return x[0] + scaler * ((x[1] + x[2]) - (x[3] + x[4]))

        def _best1(r, b):
            indices = self._sample_pop(2)
            x = r[indices.T]
            return b + scaler * (x[0] + x[1])

        def _best2(r, b):
            indices = self._sample_pop(4)
            x = r[indices.T]
            return b + scaler * ((x[0] + x[1]) - (x[2] + x[3]))

        strategies = {'best1bin': _best1,
                      # 'randtobest1bin': '_randtobest1',
                      # 'currenttobest1bin': '_currenttobest1',
                      'best2bin': _best2,
                      'rand2bin': _rand2,
                      'rand1bin': _rand1}

        assert strategy in strategies.keys()

        if "best" in strategy:
            mutation = strategies[strategy](actions, best)
        else:
            mutation = strategies[strategy](actions)

        mutation = np.clip(mutation, a_min=self.low_bound, a_max=self.high_bound)

        return mutation

    @staticmethod
    def _crossover(mutated, target, dims, cr):
        p = np.random.rand(dims)
        trial = np.where(p < cr, mutated, target)
        return trial

    def _generate_new_actions(self, actions: dict[str, np.ndarray], strategy='best2bin', mut=(0.5, 1), cr=0.7):
        flatten_actions = np.concatenate([actions[b_name] for b_name in self.b_names], axis=1)
        flatten_best = np.concatenate([self.future_actions[b_name] for b_name in self.b_names])

        mutated = self._mutate(flatten_actions, flatten_best, strategy, mut)

        trial = self._crossover(mutated=mutated, target=flatten_actions, dims=len(flatten_best), cr=cr)

        new_actions = {b_name: np.split(trial, len(self.b_names), axis=1)[i] for i, b_name in enumerate(self.b_names)}
        return new_actions

    def _update_actions(self, actions, new_actions, update_mask):
        for b_name in self.b_names:
            actions[b_name][update_mask] = new_actions[b_name][update_mask]

    def optimize(self, generations: int, tol: float = 0.01,
                 future_states: dict = None,
                 future_electricity=None, future_carbon=None
                 ):
        # initialize
        actions = self._generate_population()
        obj = self.objective_function(actions, future_states, future_electricity, future_carbon)
        best_obj_hist = [min(obj)]
        convergence = np.std(obj) / np.mean(obj)

        # optimization
        for i in range(generations):
            if convergence < tol:
                break
            new_actions = self._generate_new_actions(actions)

            new_obj = self.objective_function(new_actions)

            update_mask = new_obj < obj
            self._update_actions(actions, new_actions, update_mask)
            obj[update_mask] = new_obj[update_mask]

            self.future_actions = {b_name: actions[b_name][np.argmin(obj)].reshape(1, -1) for b_name in self.b_names}
            best_obj_hist.append(min(obj))
            convergence = np.std(obj) / np.mean(obj)
        return best_obj_hist

    def predict_mpc(self):
        # _ = self.optimize(generations)
        return [self.future_actions[b_name][0][0] for b_name in self.b_names]

    def reset_mpc(self):
        self.current_timestep = self.env.simulation_start_time_step

        # Reinitialize dictionaries
        for b, b_name in zip(self.env.buildings, self.b_names):
            self.previous_states[b_name]['non_shiftable_load'] = np.zeros(self.lookback + 1)
            self.previous_states[b_name]['solar_generation'] = np.zeros(self.lookback + 1)
            self.current_SoC[b_name] = b.observations()['electrical_storage_soc']
            self.future_states[b_name]['non_shiftable_load'] = np.zeros(self.horizon)
            self.future_states[b_name]['solar_generation'] = np.zeros(self.horizon)
            self.baseline_net_electricity[b_name] = np.zeros(self.horizon)
            self.future_actions[b_name] = np.zeros((1, self.horizon))

        # update dictionaries for initial step
        self._warmup(self.lookback)
        self._predict_future_states()
