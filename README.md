Customer Churn Prediction
🔍 Overview

This project predicts customer churn (whether a customer will leave or stay) using Logistic Regression with scaled features.

It supports:

Training a churn model on CSV data.

Saving the trained model (.pkl) for reuse.

Predicting churn for new customer data via CLI.

Automated CI/CD pipeline on GitLab with build → test → deploy → run stages.



⚙️ Installation

Clone the repo:

git clone https://gitlab.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction


Install dependencies:

pip install -r requirements.txt

📑 Dataset

The dataset (data/churn_data.csv) contains:

Age: Customer age

Tenure: Months as customer

Usage: Monthly service usage

SupportCalls: Support tickets raised

Churn: Target (Yes/No)

🚀 Usage
1️⃣ Train Model
python src/churn_predictor.py --train


➡️ Trains logistic regression model, evaluates accuracy, saves model as models/churn_model.pkl.

2️⃣ Predict Churn
python src/churn_predictor.py --predict AGE TENURE USAGE SUPPORT_CALLS


Example:

python src/churn_predictor.py --predict 40 15 250 2


➡️ Predicts if this customer will churn.

🔄 GitLab CI/CD

This project includes a 4-stage pipeline (.gitlab-ci.yml):

Build: Install dependencies, prepare environment.

Test: Train model (--train) and verify it works.

Deploy: Save trained model as artifact.

Run: Execute sample prediction (--predict 40 15 250 2).

Example .gitlab-ci.yml flow:

stages:
  - build
  - test
  - deploy
  - run


GitLab will:

Train the model automatically.

Save the trained model (.pkl).

Run a prediction as proof.
