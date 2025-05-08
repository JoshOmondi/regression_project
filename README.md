# **Housing Price Regression Project**

This project demonstrates a machine learning pipeline to predict housing prices using a **Random Forest Regressor**. The goal is to load the data, preprocess it, train a model, evaluate it, and save the trained model for future use.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Credits](#credits)

---

## **Project Overview**

This project predicts housing prices based on features such as housing features and location data. It uses the **California Housing Dataset** for training and testing, with a **Random Forest Regressor** to model the relationship between the features and the target variable (housing price). The pipeline involves the following steps:
1. **Loading the data**: Data is loaded from a CSV file.
2. **Preprocessing the data**: Features are scaled, and the dataset is split into training and testing sets.
3. **Training the model**: A Random Forest model is trained on the training set.
4. **Evaluating the model**: The model is evaluated using metrics like **Mean Squared Error (MSE)** and **R² Score**.
5. **Saving the model**: The trained model and scaler are saved for future predictions.

---

## **Installation**

To set up the project and run the model, follow these steps:

### 1. **Clone the repository**

If you haven't already, clone the project repository to your local machine:

```bash
git clone https://github.com/JoshOmondi/housing-price-regression.git
cd housing-price-regression
