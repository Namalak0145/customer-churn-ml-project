# 🧠 Customer Churn Prediction (End-to-End ML Project)

This is an industry-style, end-to-end machine learning project to predict **customer churn** using structured data from a telecom company.

Built with:
- ✅ Clean Python code
- ✅ Modular scripts
- ✅ Reproducible pipeline
- ✅ Logistic regression model
- ✅ Confusion matrix + ROC AUC evaluation

---

## 📁 Project Structure

\`\`\`
churn_prediction_project/
├── data/              # Raw dataset (CSV)
├── scripts/           # Python modules
│   ├── eda.py         # EDA & visualizations
│   ├── preprocess.py  # Data cleaning, encoding, scaling
│   ├── train_model.py # Model training & saving
│   └── evaluate.py    # Evaluation & metrics
├── models/            # Saved model (pkl file)
├── main.py            # Full pipeline runner
├── README.md          # You're here!
\`\`\`

---

## 🔧 Tools Used

- Python 🐍
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Joblib
- VS Code
- GitHub

---

## 🚀 How to Run

1. Clone this repo
2. Create a virtual environment and activate it:
    \`\`\`bash
    python -m venv venv
    venv\Scripts\activate
    \`\`\`
3. Install dependencies:
    \`\`\`bash
    pip install -r requirements.txt
    \`\`\`
4. Download the dataset from [Kaggle Telco Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
   Place it in the \`data/\` folder as \`telco_churn.csv\`

5. Run the full pipeline:
    \`\`\`bash
    python main.py
    \`\`\`

---

## 📊 Sample Output

Model Accuracy: ~78%  
Includes confusion matrix, ROC-AUC curve, and full classification report.

---

## 🌟 Future Enhancements

- Add a Streamlit-based frontend
- Save predictions to a dashboard
- Try other models (XGBoost, SVM, RandomForest)

---

## 🙌 Author

**Krishna Prasad**  
_ML Learner • Python Enthusiast • Open to Opportunities_  
[GitHub](https://github.com/Namalak0145)

---

## 📜 License

This project is open-source and free to use under the MIT License.