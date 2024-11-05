# particle_swarm_supervised.py

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class ParticleSwarmSupervisedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_particles=20, n_iterations=10, c1=1.5, c2=1.5, w=0.5, random_state=None):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.random_state = random_state
        self.selected_indices_ = None  

    def fit(self, X, y):
        np.random.seed(self.random_state)
        
        X = X.values if hasattr(X, 'values') else X  
        y = y.values if hasattr(y, 'values') else y
        
        n_features = X.shape[1]
        
        classifiers = [
            LogisticRegression(max_iter=1000, random_state=self.random_state),
            SVC(probability=True, random_state=self.random_state),
            DecisionTreeClassifier(random_state=self.random_state),
            RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        ]
        
        def fitness(position):
            selected_features = X[:, position > 0.5]
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
        
        positions = np.random.rand(self.n_particles, n_features)
        velocities = np.random.rand(self.n_particles, n_features) * 0.1

        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([fitness(pos) for pos in positions])
        global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
        global_best_score = np.max(personal_best_scores)
        
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
                cognitive_velocity = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.c2 * r2 * (global_best_position - positions[i])
                velocities[i] = self.w * velocities[i] + cognitive_velocity + social_velocity
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], 0, 1)
                score = fitness(positions[i])
                if score > personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = score
                if score > global_best_score:
                    global_best_position = positions[i]
                    global_best_score = score
            print(f"Iteration {iteration + 1} - Best Fitness: {global_best_score}")
        
        self.selected_indices_ = np.where(global_best_position > 0.5)[0]
        print("Selected Feature Indices:", self.selected_indices_)
        return self

    def transform(self, X):
        if self.selected_indices_ is None:
            raise RuntimeError("You must fit the selector before transforming data!")
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)
