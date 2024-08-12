# Loan Default Prediction using Decision Trees and Naive Bayes Classifiers

This project involves the implementation and comparison of Machine Learning (ML) models—specifically Decision Trees and Naive Bayes classifiers—to predict whether a new customer is likely to repay a loan using the Lending Club dataset. The project covers model building, testing, hyperparameter tuning, and performance evaluation on both synthetic and real-world datasets.

## Project Structure

The project consists of the following components:

- **`decision_tree.py`**: Contains the implemented functions for building and evaluating Decision Tree models.
- **`naive_bayes.py`**: Contains the implemented functions for training and predicting with Naive Bayes classifiers.
- **`experiments.ipynb`**: Jupyter notebook used for running tests, debugging, and conducting experiments on both synthetic and real-world datasets.
- **`dataset/`**: Directory containing the synthetic datasets and the Lending Club dataset.

## Implementation Overview

### 1. Decision Tree

In `decision_tree.py`, the following functions have been implemented:

- **`Entropy(y)`**: 
  - Computes the entropy of the distribution over class labels given a dataset with labels `y`.
  - This function is essential for determining the purity of a split in the decision tree, a crucial part of the ML model.

- **`optimal_split(X, y, feature)`**: 
  - Sorts data points according to the value of a selected feature.
  - Evaluates the information gain for hypothetical split thresholds, identifying the optimal split for the decision tree.
  - This function is used during the tree-building process to determine the best splits, optimizing the ML model's performance.

### 2. Naive Bayes

In `naive_bayes.py`, the following functions have been implemented:

- **`train_nb(X, y)`**:
  - Trains the Naive Bayes model using maximum likelihood estimates.
  - Computes the prior probabilities, and the mean and covariance for the Gaussian likelihood for each class.
  - This function is central to developing the Naive Bayes ML model.

- **`predict_nb(X)`**:
  - Computes the probability of each input belonging to each class.
  - This function allows for making predictions with the trained Naive Bayes model, facilitating the ML process.

### 3. Debugging and Analysis

- **Testing**: 
  - The `experiments.ipynb` notebook was used to run provided and additional test cases during the development process.
  - These tests helped in identifying and fixing bugs in the initial implementation of the ML models.

- **Experiments**:
  - After successful debugging, experiments were conducted on the Lending Club dataset.
  - The Decision Tree model was tuned by adjusting hyperparameters such as maximum depth, minimum samples per leaf, and minimum entropy required to split a node, optimizing the ML model.

### 4. Final Evaluation

- **Model Comparison**:
  - After tuning the Decision Tree, its performance was compared with that of the Naive Bayes classifier.
  - Performance metrics such as accuracy, precision, recall, and F1-score were used to evaluate and compare the ML models.

## Results and Insights

- **Decision Tree**:
  - **Train accuracy**: 79.42% ± 0.21%
  - **Validation accuracy**: 79.30% ± 0.37%
  - **Test accuracy**: 79.01% ± 0.26%
  - The Decision Tree model performed consistently across the training, validation, and test sets, with accuracy around 79%.

- **Naive Bayes**:
  - **Train accuracy**: 67.86%
  - **Validation accuracy**: 67.70%
  - **Test accuracy**: 67.37%
  - The Naive Bayes model showed lower accuracy compared to the Decision Tree, with performance around 67% across all datasets.

## Conclusion

Based on the results and analysis, the Decision Tree model outperformed the Naive Bayes model. After tuning the hyperparameters, the accuracy for the Decision Tree was significantly better than that of the Naive Bayes model. Therefore, I would choose the Decision Tree over Naive Bayes for this particular task of predicting loan repayment.

## Dataset

The real-world dataset used in this project is the Lending Club dataset, which includes 21 features per data point. These features represent various aspects of a customer’s financial background, loan amount, location, etc. The primary task was to predict the likelihood of a customer repaying their loan, a typical problem in ML.

## Usage

1. **Clone the Repository**: 
    ```bash
    git clone https://github.com/Maheer1207/Loan-Default-Predictor.git
    cd Loan-Default-Predictor
    ```

2. **Install Dependencies**: 
   - Ensure that all required Python packages are installed by running:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Experiments**:
   - Open `experiments.ipynb` and follow the steps to run the final experiments.
   - Compare the performance of the Decision Tree and Naive Bayes classifiers on the Lending Club dataset.

## Contributing

If you wish to contribute to this project, please fork the repository, create a new branch, and submit a pull request with your proposed changes.
