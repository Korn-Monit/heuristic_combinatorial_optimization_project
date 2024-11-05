# simulated_annealing.py
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

class SimulatedAnnealingSupervisedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, initial_temperature=1000, cooling_rate=0.95, n_iterations=100, random_state=42):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.selected_indices_ = None
        self.execution_time_ = None
        self.auc_scores_ = None
        self.auc_mean_ = None
        self.auc_std_ = None

    def fit(self, X, y):
        start_time = time.time()
        np.random.seed(self.random_state)
        n_features = X.shape[1]

        classifiers = [
            LogisticRegression(max_iter=1000, random_state=self.random_state),
            SVC(probability=True, random_state=self.random_state),
            DecisionTreeClassifier(random_state=self.random_state),
            RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        ]

        def fitness(chromosome):
            selected_features = X[:, chromosome == 1]
            if selected_features.shape[1] == 0:
                return 0  

            auc_scores = []
            for clf in classifiers:
                pipeline = make_pipeline(StandardScaler(), clf)
                try:
                    scores = cross_val_score(pipeline, selected_features, y, cv=5, scoring='roc_auc_ovr')
                    auc_scores.append(np.mean(scores))
                except Exception as e:
                    auc_scores.append(0)
            return np.mean(auc_scores)

        current_solution = np.random.randint(2, size=n_features)
        if not current_solution.any():
            current_solution[np.random.randint(0, n_features)] = 1
        current_fitness = fitness(current_solution)

        best_solution = current_solution.copy()
        best_fitness = current_fitness

        temperature = self.initial_temperature

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

        selected_features = X[:, self.selected_indices_]
        auc_scores = []
        for clf in classifiers:
            pipeline = make_pipeline(StandardScaler(), clf)
            try:
                scores = cross_val_score(pipeline, selected_features, y, cv=5, scoring='roc_auc_ovr')
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
