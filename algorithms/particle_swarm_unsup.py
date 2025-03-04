import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
class ParticleSwarmUnsupervisedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_particles=20,
        n_iterations=10,
        w=0.729,
        c1=1.49445,
        c2=1.49445,
        alpha=0.5,
        beta=0.5,
        gamma=0.01,
        random_state=None,
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.random_state = random_state
        self.selected_indices_ = None  

    def fit(self, X, y=None):
        np.random.seed(self.random_state)
        X = X.values if hasattr(X, 'values') else X  

        n_features = X.shape[1]

        def fitness(position):
            selected_features = X[:, position == 1]
            n_s = selected_features.shape[1]  

            if n_s == 0:
                return -np.inf

            std_devs = np.std(selected_features, axis=0)
            non_constant_indices = np.where(std_devs > 0)[0]
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

            variance_score = np.mean(np.var(selected_features, axis=0))

            fitness_score = self.alpha * variance_score - self.beta * mi_score_avg - self.gamma * n_s
            return fitness_score

        positions = np.random.randint(2, size=(self.n_particles, n_features))  
        velocities = np.random.uniform(low=-1, high=1, size=(self.n_particles, n_features))

        for position in positions:
            if not position.any():
                position[np.random.randint(0, n_features)] = 1

        pbest_positions = positions.copy()
        pbest_fitnesses = np.array([fitness(pos) for pos in positions])

        gbest_index = np.argmax(pbest_fitnesses)
        gbest_position = pbest_positions[gbest_index].copy()
        gbest_fitness = pbest_fitnesses[gbest_index]

        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                r1 = np.random.rand(n_features)
                r2 = np.random.rand(n_features)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (pbest_positions[i] - positions[i])
                    + self.c2 * r2 * (gbest_position - positions[i])
                )

                sigmoid = 1 / (1 + np.exp(-velocities[i]))

                rand_vals = np.random.rand(n_features)
                positions[i] = np.where(rand_vals < sigmoid, 1, 0)

                if not positions[i].any():
                    positions[i][np.random.randint(0, n_features)] = 1

                fitness_value = fitness(positions[i])

                if fitness_value > pbest_fitnesses[i]:
                    pbest_positions[i] = positions[i].copy()
                    pbest_fitnesses[i] = fitness_value

            gbest_index = np.argmax(pbest_fitnesses)
            if pbest_fitnesses[gbest_index] > gbest_fitness:
                gbest_position = pbest_positions[gbest_index].copy()
                gbest_fitness = pbest_fitnesses[gbest_index]

            print(f"Iteration {iteration + 1} - Best Fitness: {gbest_fitness}")

        self.selected_indices_ = np.where(gbest_position == 1)[0]
        print("Selected Feature Indices:", self.selected_indices_)
        return self

    def transform(self, X):
        if self.selected_indices_ is None:
            raise RuntimeError("You must fit the selector before transforming data!")
        X = X.values if hasattr(X, 'values') else X
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)
