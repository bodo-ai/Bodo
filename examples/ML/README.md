# Machine Learning Examples

This directory contains various machine learning examples using Bodo. These examples demonstrate how to leverage Bodo for accelerating machine learning model training and prediction performance. 
  
## Examples

### 1. Accelerating ML credit card fraud detection model prediction performance.

- **File**: `credit-card-fraud.ipynb` 

- **Description:** This notebook builds a machine learning model for credit card fraud detection. It focuses on handling imbalanced data using sub-sampling and SMOTE (Synthetic Minority Over-sampling Technique) and uses a Random Forest Classifier from scikit-learn, all accelerated with Bodo.

- **Bodo Benefits:** This tackles a very common and important ML problem (fraud detection) and demonstrates how Bodo can be used with techniques like oversampling to improve model performance on imbalanced datasets. It's another end-to-end example of using Bodo for a real-world ML task.

### 2. Accelerating large data set scaling for ML flight cancelation predictions.

- **File**: `flight-cancelation-prediction.ipynb`

- **Description:** This notebook demonstrates how to implement and test a Random Forest Classifier and Logistic Regression to predict flight cancellations.

- **Bodo Benefits:** This is a robust ML example showcasing how Bodo can be used to scale common scikit-learn models to larger datasets.

### 3. Accelerating data processing and model training for Pandas and scikit-learn predictive modeling.

- **File**: `taxi-tips-prediction.ipynb`

- **Description**: This notebook builds a machine learning model to predict taxi tips in New York City, using the NYC Yellow Cab trip record data. It demonstrates data preprocessing, feature engineering, model training (using scikit-learn), and evaluation, all within a Bodo-accelerated environment.

- **Bodo Benefits:** This provides a complete ML workflow example, showcasing Bodo's ability to accelerate data preparation and model training using popular libraries like Pandas and scikit-learn.  It shows how to apply Bodo to a practical predictive modeling task.
