# 🏍️ Customer Purchase Prediction using Classical Machine Learning

This project demonstrates a simple classical Machine Learning pipeline that predicts whether a customer will make a purchase based on their **Age** and **Salary**. It leverages the popular `K-Nearest Neighbors (KNN)` algorithm and is implemented in Python using `scikit-learn`.

---

## 🚀 Project Overview

Businesses often seek to predict which potential customers are likely to purchase their products. In this project:

✅ We build a model that takes **Age** and **Salary** as input features.
✅ The model predicts whether a customer will make a **purchase** or not.
✅ The model is trained on a cleaned dataset: `cleaned_purchase_data.csv`.

---

## 📂 Project Structure

```
├── cleaned_purchase_data.csv
├── purchase_prediction_model.py  # Main Python script
├── README.md
└── requirements.txt
```

---

## ⚙️ How It Works

1. **Load the dataset**
   Data is loaded from a CSV file that contains customer information.

2. **Preprocess the data**
   Features are scaled using `StandardScaler` to improve model performance.

3. **Train/test split**
   Data is split into training and testing sets (80/20 split).

4. **Model training**
   A **K-Nearest Neighbors (KNN)** classifier is trained on the training set.

5. **Model evaluation**
   The model predicts on the test set and outputs both:

   * The predicted labels
   * The predicted probabilities

---

## 📊 Requirements

* Python 3.x
* pandas
* numpy
* scikit-learn

You can install the required packages using:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
pandas
numpy
scikit-learn
```

---

## 💻 Usage

1. Clone the repository:

```bash
git clone https://github.com/Jonathan463/customer-purchase-prediction.git
cd customer-purchase-prediction
```

2. Place your `cleaned_purchase_data.csv` file in the project directory.

3. Run the model:

```bash
python purchase_prediction_model.py
```

---

## 📈 Future Improvements

* ✅ Add cross-validation to better evaluate model performance.
* ✅ Try other classification algorithms (Logistic Regression, Random Forest, SVM).
* ✅ Perform hyperparameter tuning for KNN.
* ✅ Visualize decision boundaries.
* ✅ Deploy the model via API (FastAPI or Flask).
* ✅ Add unit tests.

---

## 🤝 Contributing

Feel free to fork this project and submit a pull request with improvements!
Contributions, issues and feature requests are welcome.

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
