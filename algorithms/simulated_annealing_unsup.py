# simulated_annealing_unsup.py

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

class SimulatedAnnealingUnsupervisedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        initial_temperature=1000,
        cooling_rate=0.95,
        n_iterations=100,
        alpha=0.5,
        beta=0.5,
        gamma=0.01,
        random_state=None,
    ):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.random_state = random_state
        self.selected_indices_ = None 

    def fit(self, X, y=None):
        np.random.seed(self.random_state)
        X = X.values if hasattr(X, 'values') else X  

        n_features = X.shape[1]
        temperature = self.initial_temperature

        def fitness(chromosome):
            selected_features = X[:, chromosome == 1]
            n_s = selected_features.shape[1]  

            if n_s == 0:
                return -np.inf

            variances = np.var(selected_features, axis=0)
            non_constant_indices = np.where(variances > 0)[0]
            selected_features = selected_features[:, non_constant_indices]
            n_s = selected_features.shape[1]
            if n_s == 0:
                return -np.inf

            unique_bins = [len(np.unique(selected_features[:, i])) for i in range(n_s)]
            adjusted_bins = [max(2, min(10, ub)) for ub in unique_bins]
            discretizer = KBinsDiscretizer(n_bins=adjusted_bins, encode='ordinal', strategy='uniform')
            selected_features_discrete = discretizer.fit_transform(selected_features)

            max_pairs = 1000  
            indices = np.triu_indices(n_s, k=1)
            all_pairs = list(zip(indices[0], indices[1]))
            if len(all_pairs) > max_pairs:
                sampled_indices = np.random.choice(len(all_pairs), max_pairs, replace=False)
                sampled_pairs = [all_pairs[i] for i in sampled_indices]
            else:
                sampled_pairs = all_pairs

            mi_scores = [
                mutual_info_score(selected_features_discrete[:, i], selected_features_discrete[:, j])
                for i, j in sampled_pairs
            ]
            mi_score_avg = np.mean(mi_scores) if mi_scores else 0

            variance_score = np.mean(variances[non_constant_indices])

            fitness_score = self.alpha * variance_score - self.beta * mi_score_avg - self.gamma * n_s
            return fitness_score

        current_solution = np.random.randint(2, size=n_features)
        if not current_solution.any():
            current_solution[np.random.randint(0, n_features)] = 1
        current_fitness = fitness(current_solution)

        best_solution = current_solution.copy()
        best_fitness = current_fitness

        for iteration in range(self.n_iterations):
            neighbor_solution = current_solution.copy()
            flip_index = np.random.randint(0, n_features)
            neighbor_solution[flip_index] = 1 - neighbor_solution[flip_index]
            if not neighbor_solution.any():
                neighbor_solution[np.random.randint(0, n_features)] = 1

            neighbor_fitness = fitness(neighbor_solution)

            delta_fitness = neighbor_fitness - current_fitness
            if delta_fitness > 0:
                acceptance_probability = 1.0
            else:
                acceptance_probability = np.exp(delta_fitness / temperature)

            if acceptance_probability > np.random.rand():
                current_solution = neighbor_solution.copy()
                current_fitness = neighbor_fitness

                if current_fitness > best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness

            temperature *= self.cooling_rate

            print(f"Iteration {iteration + 1}/{self.n_iterations} - Current Fitness: {current_fitness:.4f}, Best Fitness: {best_fitness:.4f}, Temperature: {temperature:.4f}")

        self.selected_indices_ = np.where(best_solution == 1)[0]

        print("Selected Feature Indices:", self.selected_indices_)
        return self

    def transform(self, X):
        if self.selected_indices_ is None:
            raise RuntimeError("You must fit the selector before transforming data!")
        X = X.values if hasattr(X, 'values') else X
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)
