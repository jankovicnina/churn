# Churn Prediction Project

## Project Overview
This project implements a machine learning pipeline to predict customer churn for a music streaming service. The goal is to identify users who are likely to cancel their subscription within a 10-day prediction window based on their historical activity data.

# File Structure and Purpose

1. Initial Data Exploration
 
The first notebook/script titled `1_data_exploration.ipnyb` focused on understanding the fundamental characteristics of the dataset. We conducted comprehensive analysis to comprehend the data structure and schema by examining data types, dimensions, and variable distributions. This phase involved loading and inspecting the raw data, categorizing columns as categorical or continuous, assessing unique value counts, and performing correlation analysis between numerical features. The initial visualizations and data profiling established the groundwork for identifying data quality issues and setting up subsequent analytical steps. The exploratory work from this stage was instrumental in shaping our overall feature engineering strategy and informed our understanding of the underlying patterns within the user activity data.

2. Feature Analysis

The second notebook, `2_time_analysis.ipynb`, builds upon the initial exploration by focusing on time-based user behavior analysis and comprehensive feature engineering. We analyzed user activity patterns in the 10 days preceding churn (for churned users) or the last 10 days of available data (for retained users). This notebook implemented systematic feature extraction across multiple dimensions:

- Temporal Analysis: Calculated daily activity metrics including songs played, positive interactions (likes, adds), negative interactions (dislikes, errors), and help/support usage
- Behavioral Segmentation: Created functions to analyze individual user patterns and aggregate statistics across user groups
- Multi-Dimensional Features: Engineered features spanning activity frequency, engagement quality, music diversity, session patterns, and temporal trends
- Comparative Analysis: Examined differences between churned and retained users across all extracted features

The feature analysis phase produced a rich dataset of behavioral indicators that serve as the foundation for predictive modeling, with particular emphasis on identifying patterns that precede churn events.

3. `3.0_deep_learning.py` - Initial Deep Learning Approach

This script represents our first modeling attempt using deep learning techniques to predict user churn. We implemented an LSTM (Long Short-Term Memory) neural network architecture designed to capture sequential patterns in user behavior over time. The model processed 20-day windows of daily user activity data, treating the churn prediction problem as a time-series classification task. Our architecture featured a 2-layer LSTM network with dropout regularization to prevent overfitting and improve generalization.

After extensive experimentation, we decided to move away from this deep learning approach for several key reasons. First, the available training data proved insufficient for deep neural networks to demonstrate their full potential, as these models typically require large datasets to excel. Second, we encountered challenges with model interpretability—while tree-based methods offer transparent decision paths, the LSTM's internal workings remained largely a "black box," making it difficult to explain predictions to stakeholders. Third, and most critically, the LSTM model underperformed compared to simpler gradient boosting methods, achieving lower predictive accuracy despite its computational complexity. Finally, the substantial computational resources required for training and tuning the LSTM did not justify the marginal performance gains, if any, over more efficient alternatives.

Despite not being adopted for production, this deep learning experiment provided valuable insights. It confirmed that complex sequential modeling was not necessary for this particular churn prediction task, and it helped establish baseline expectations for what advanced techniques could achieve with our dataset. The learnings from this approach informed our subsequent modeling decisions and reinforced our commitment to practical, interpretable, and computationally efficient solutions.

4. Production Pipeline (`3.1_final_model.py`) - Primary Model

This script represents our main production-ready churn prediction pipeline, implementing a machine learning workflow. At its core, we employ an ensemble learning approach with seed averaging across five different random seeds to enhance model stability and reduce variance. The pipeline addresses class imbalance through strategic undersampling of non-churners, achieving a controlled 35% churn rate in training data while maintaining representative patterns. Our temporal analysis utilizes a sliding window mechanism with a 20-day observation period and 7-day stride, allowing for continuous monitoring and early detection of churn signals.

The feature engineering process captures multiple dimensions of user behavior, including engagement metrics like session counts and total listening time, temporal patterns such as time-of-day activity distributions and weekend usage ratios, detailed action analysis of specific page interactions, trend features comparing short-term versus long-term engagement changes, and demographic factors including account tier and user tenure. Technical innovations in the pipeline include vectorized time-slicing using binary search for efficient window extraction, adaptive thresholding that optimizes classification boundaries based on precision-recall trade-offs, special logic for handling inactive users with no recent activity, and mutual information-based selection of the top 20 most predictive features.

The model architecture centers on XGBoost with histogram-based tree methods for computational efficiency, combined into a five-model ensemble trained with different random seeds. The training process employs 3-fold cross-validation with Optuna optimization across 150 trials to identify optimal hyperparameters. For prediction, the pipeline averages probabilities across all ensemble members, creating a robust and well-calibrated churn risk score that balances predictive power with operational efficiency for real-time deployment.
