# genetic_algorithm.py
import numpy as np
import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class GeneticAlgorithmSupervisedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_population=20, n_generations=10, mutation_rate=0.1, random_state=42):
        self.n_population = n_population
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.selected_indices_ = None
        self.execution_time_ = None
        self.auc_scores_ = None  # Store AUC scores for each classifier
        self.auc_mean_ = None
        self.auc_std_ = None

    def fit(self, X, y):
        start_time = time.time()  # Start timing
        np.random.seed(self.random_state)  # For reproducibility
        n_features = X.shape[1]

        # List of classifiers to evaluate
        classifiers = [
            LogisticRegression(max_iter=1000, random_state=self.random_state),
            SVC(probability=True, random_state=self.random_state),
            DecisionTreeClassifier(random_state=self.random_state),
            RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        ]

        # Parameters for the Genetic Algorithm
        n_population = self.n_population  # Number of chromosomes in the population
        n_generations = self.n_generations  # Number of generations to evolve
        mutation_rate = self.mutation_rate  # Probability of mutation

        # Fitness function to evaluate each chromosome

        def fitness(chromosome):
            # Select features based on chromosome
            selected_features = X[:, chromosome == 1]
            if selected_features.shape[1] == 0:
                return 0  # Avoid empty feature selection

            auc_scores = []
            for clf in classifiers:
                # Create a pipeline with scaling and the classifier
                pipeline = make_pipeline(StandardScaler(), clf)
                try:
                    scores = cross_val_score(pipeline, selected_features, y, cv=5, scoring='roc_auc_ovr')
                    auc_scores.append(np.mean(scores))
                except Exception as e:
                    auc_scores.append(0)  # If classifier fails, assign score 0
            return np.mean(auc_scores)  # Return average AUC across classifiers

        # Initialize the population with random feature selection
        population = np.random.randint(2, size=(n_population, n_features))
        # Ensure that no chromosome is all zeros
        for chromosome in population:
            if not chromosome.any():
                chromosome[np.random.randint(0, n_features)] = 1

        # Evaluate fitness of the initial population
        fitness_scores = np.array([fitness(chromosome) for chromosome in population])

        # Genetic Algorithm
        for generation in range(n_generations):
            # Selection: Select parents based on fitness
            parents = population[np.argsort(fitness_scores)][-n_population//2:]

            # Crossover: Generate new offspring by combining pairs of parents
            offspring = []
            for _ in range(n_population // 2):
                parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
                crossover_point = np.random.randint(1, n_features - 1)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                offspring.append(child)
            offspring = np.array(offspring)

            # Mutation: Randomly flip bits in some offspring
            mutation_mask = np.random.rand(*offspring.shape) < mutation_rate
            offspring = np.logical_xor(offspring, mutation_mask).astype(int)
            # Ensure that mutated offspring are not all zeros
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

        # Evaluate AUC scores across classifiers for the best chromosome
        selected_features = X[:, self.selected_indices_]
        auc_scores = []
        for clf in classifiers:
            try:
                scores = cross_val_score(clf, selected_features, y, cv=5, scoring='roc_auc_ovr')
                auc_scores.append(np.mean(scores))
            except Exception as e:
                auc_scores.append(0)
        self.auc_scores_ = auc_scores
        self.auc_mean_ = np.mean(auc_scores)
        self.auc_std_ = np.std(auc_scores)

        end_time = time.time()
        self.execution_time_ = end_time - start_time

        return self

    def transform(self, X):
        return X[:, self.selected_indices_]
