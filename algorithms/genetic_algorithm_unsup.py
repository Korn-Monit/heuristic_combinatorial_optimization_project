# genetic_algorithm_unsup.py

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

class GeneticAlgorithmUnsupervisedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_population=20,
        n_generations=10,
        mutation_rate=0.1,
        alpha=0.5,
        beta=0.5,
        gamma=0.01,
        random_state=None,
    ):
        self.n_population = n_population
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.random_state = random_state
        self.selected_indices_ = None  # To store selected feature indices

    def fit(self, X, y=None):
        np.random.seed(self.random_state)
        X = X.values if hasattr(X, 'values') else X  # Ensure NumPy array

        n_features = X.shape[1]

        def fitness(chromosome):
            # Select features based on the chromosome
            selected_features = X[:, chromosome == 1]
            n_s = selected_features.shape[1]  # Number of selected features

            # Avoid empty selection
            if n_s == 0:
                return -np.inf

            # Remove constant features
            non_constant_indices = np.where(np.std(selected_features, axis=0) > 0)[0]
            selected_features = selected_features[:, non_constant_indices]
            n_s = selected_features.shape[1]
            if n_s == 0:
                return -np.inf

            # Adjust the number of bins for discretization
            unique_bins = [len(np.unique(selected_features[:, i])) for i in range(n_s)]
            adjusted_bins = [max(2, min(10, ub)) for ub in unique_bins]
            discretizer = KBinsDiscretizer(n_bins=adjusted_bins, encode='ordinal', strategy='uniform')
            selected_features_discrete = discretizer.fit_transform(selected_features)

            # Limit the number of mutual information computations
            max_pairs = 1000  # Limit the number of pairs
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

            # Compute variance score
            variance_score = np.mean(np.var(selected_features, axis=0))

            # Compute the fitness score
            fitness_score = self.alpha * variance_score - self.beta * mi_score_avg - self.gamma * n_s
            return fitness_score

        # Initialize the population with random feature selection
        population = np.random.randint(2, size=(self.n_population, n_features))
        # Ensure no chromosome is all zeros
        for chromosome in population:
            if not chromosome.any():
                chromosome[np.random.randint(0, n_features)] = 1

        # Evaluate fitness of the initial population
        fitness_scores = np.array([fitness(chromosome) for chromosome in population])

        # Genetic Algorithm
        for generation in range(self.n_generations):
            # Selection: Select parents based on fitness
            parents = population[np.argsort(fitness_scores)][-self.n_population // 2:]

            # Crossover: Generate new offspring by combining pairs of parents
            offspring = []
            for _ in range(self.n_population // 2):
                parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
                crossover_point = np.random.randint(1, n_features - 1)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                offspring.append(child)
            offspring = np.array(offspring)

            # Mutation: Randomly flip bits in some offspring
            mutation_mask = np.random.rand(*offspring.shape) < self.mutation_rate
            offspring = np.logical_xor(offspring, mutation_mask).astype(int)
            # Ensure mutated offspring are not all zeros
            for chromosome in offspring:
                if not chromosome.any():
                    chromosome[np.random.randint(0, n_features)] = 1

            # Create new population by combining parents and offspring
            population = np.vstack((parents, offspring))

            # Evaluate fitness of the new population
            fitness_scores = np.array([fitness(chromosome) for chromosome in population])

            # Best fitness in the current generation
            best_fitness = np.max(fitness_scores)
            print(f"Generation {generation + 1} - Best Fitness: {best_fitness}")

        # Final selection of the best chromosome
        best_index = np.argmax(fitness_scores)
        best_chromosome = population[best_index]
        self.selected_indices_ = np.where(best_chromosome == 1)[0]

        print("Selected Feature Indices:", self.selected_indices_)
        return self

    def transform(self, X):
        if self.selected_indices_ is None:
            raise RuntimeError("You must fit the selector before transforming data!")
        X = X.values if hasattr(X, 'values') else X
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)
