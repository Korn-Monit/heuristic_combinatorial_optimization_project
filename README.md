# **Feature Selection in Machine Learning using Metaheuristics**

## **Project Overview**
This project tackles the **Feature Selection Problem**, a **combinatorial optimization challenge**, using **metaheuristics** to select the most informative features from **genomic datasets**. Due to the high dimensionality of genomic data, feature selection is critical for improving **model accuracy, computational efficiency, and interpretability**.

Two optimization settings were explored:

1. **Unsupervised Setting** - Feature selection based on **mutual information**, minimizing redundancy without using labels.
2. **Supervised Setting** - Feature selection based on **AUC score**, directly assessing predictive power with classifiers.

The project compares the efficiency of three metaheuristic algorithms: **Genetic Algorithm (GA), Particle Swarm Optimization (PSO), and Simulated Annealing (SA)**.

---

## **Dataset**
We use gene expression data obtained from the **Gene Expression Omnibus (GEO)**. The dataset consists of:

- **CSV Format**
- The **first column** represents class labels.
- The **remaining columns** contain gene expression values.

---

## **Objectives**
- Implement **three metaheuristic algorithms** for feature selection.
- Compare performance in **supervised and unsupervised settings**.
- Evaluate **AUC scores** using classifiers (**Decision Tree, Random Forest, SVC, Logistic Regression**).
- Measure and compare the **execution time** of different approaches.
- Provide a **script** that automates feature selection and evaluation.

---

<!-- ## **Project Structure** -->
