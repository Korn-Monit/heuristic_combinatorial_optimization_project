import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from algorithms.genetic_algorithm_sup import GeneticAlgorithmSupervisedFeatureSelector
from algorithms.particle_swarm_supervised import ParticleSwarmSupervisedFeatureSelector
from algorithms.genetic_algorithm_unsup import GeneticAlgorithmUnsupervisedFeatureSelector
from algorithms.particle_swarm_unsup import ParticleSwarmUnsupervisedFeatureSelector
from algorithms.simulated_annealing_sup import SimulatedAnnealingSupervisedFeatureSelector
from algorithms.simulated_annealing_unsup import SimulatedAnnealingUnsupervisedFeatureSelector

def load_data(data_path):
    df = pd.read_csv(data_path)
    labelEncoder = LabelEncoder()
    df['class_encoded'] = labelEncoder.fit_transform(df['Class'])
    df.drop(['Class'], axis=1, inplace=True)
    X = df.drop(columns=['class_encoded']).values
    y = df['class_encoded'].values
    return train_test_split(X, y, test_size=0.3, random_state=42)

def main():
    data_path = 'dataset.csv'  
    output_path = 'result'  
    os.makedirs(output_path, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data(data_path)
    algorithms = {
        'supervised_genetic_algorithm': GeneticAlgorithmSupervisedFeatureSelector(),
        'supervised_particle_swarm': ParticleSwarmSupervisedFeatureSelector(),
        'unsupervised_genetic_algorithm': GeneticAlgorithmUnsupervisedFeatureSelector(),
        'unsupervised_particle_swarm': ParticleSwarmUnsupervisedFeatureSelector(),
        'supervised_simulated_annealing': SimulatedAnnealingSupervisedFeatureSelector(),
        'unsupervised_simulated_annealing': SimulatedAnnealingUnsupervisedFeatureSelector()
    }
    classifiers = [
        LogisticRegression(max_iter=5000),
        SVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100)
    ]

    for algorithm_name, selector in algorithms.items():
        print(f"\nRunning feature selection using {algorithm_name}...")

        start_fs_time = time.time()
        selector.fit(X_train, y_train)
        end_fs_time = time.time()
        feature_selection_time = end_fs_time - start_fs_time

        print(f"Feature selection completed in {feature_selection_time:.4f} seconds.")
        print(f"Selected features indices: {selector.selected_indices_}")

        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)

        auc_scores = []
        execution_times = []

        print("\nStarting evaluation of classifiers...")
        for clf in classifiers:
            clf = clone(clf)

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classification', clf)
            ])

            print(f"\nEvaluating classifier: {clf.__class__.__name__}")
            start_time = time.time()

            cv_scores = cross_val_score(
                pipeline,
                X_train_selected,
                y_train,
                cv=5,
                scoring='roc_auc_ovr'
            )

            end_time = time.time()
            evaluation_time = end_time - start_time

            auc_mean = np.mean(cv_scores)
            auc_std = np.std(cv_scores)

            auc_scores.append({
                "classifier": clf.__class__.__name__,
                "auc_mean": auc_mean,
                "auc_std": auc_std
            })
            execution_times.append({
                "classifier": clf.__class__.__name__,
                "evaluation_time": evaluation_time
            })

            print(f"AUC Mean: {auc_mean:.4f}, AUC Std: {auc_std:.4f}, Evaluation Time: {evaluation_time:.4f} seconds")

        # Save results for the current algorithm
        result_file = os.path.join(output_path, f"results_{algorithm_name}.json")
        with open(result_file, "w") as f:
            json.dump({
                "algorithm": algorithm_name,
                "feature_selection_time": feature_selection_time,
                "auc_scores": auc_scores,
                "evaluation_times": execution_times,
                "selected_features": selector.selected_indices_.tolist()
            }, f, indent=4)

        print(f"\nResults for {algorithm_name} saved to {result_file}")

if __name__ == "__main__":
    main()
