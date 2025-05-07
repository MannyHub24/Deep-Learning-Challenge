
# Alphabet Soup Charity — Deep Learning Challenge

## Overview

The purpose of this project is to develop a deep learning model that can predict the success of applicants applying for funding from the Alphabet Soup nonprofit foundation. By using historical data on over 34,000 applications, we trained and optimized a binary classifier using TensorFlow and Keras.

---

## Files

- `AlphabetSoupCharity.h5`: Trained baseline model.
- `AlphabetSoupCharity_Optimization.h5`: Tuned and optimized model.
- `AlphabetSoupCharity.ipynb`: Notebook used to preprocess data, train, and evaluate the baseline model.
- `AlphabetSoupCharity_Optimization.ipynb`: Notebook used to optimize and tune the model using Keras Tuner.
- `AlphabetSoup_Model_Report.docx`: Final analysis report.

---

## Technologies

- Python
- Pandas
- TensorFlow/Keras
- scikit-learn
- Keras Tuner
- Google Colab

---

## Preprocessing

- Dropped non-predictive ID columns: `EIN`, `NAME`
- Encoded categorical features with `pd.get_dummies()`
- Grouped rare categories into "Other"
- Scaled features using `StandardScaler`
- Split dataset into training and testing sets (75/25 split)

---

## Model Architecture

### Baseline Model
- Input Layer: 43 features
- Hidden Layer 1: 80 neurons, ReLU
- Hidden Layer 2: 30 neurons, ReLU
- Output Layer: 1 neuron, Sigmoid
- Accuracy: ~72.8%

### Optimized Model
- Used Keras Tuner with Hyperband to explore:
  - Number of hidden layers and neurons
  - Activation functions: ReLU, Tanh
  - Dropout rates: 0.1 to 0.5
- Best accuracy: ~73%

---

## Results

Although the optimized model did not exceed the 75% target, it demonstrated solid predictive performance and incorporated at least three model tuning strategies.

---

## Recommendations

For future improvement, consider testing alternative models such as Random Forests or XGBoost, which are often more effective with tabular data.

## Author

Manuel Guevara
