# Heart Disease Classification

I built this notebook end to end to study the Heart Failure Prediction dataset and compare several classifiers, including hard and soft voting. It walks through data checks, preprocessing, cross-validation, and final model evaluation.

This is a small educational dataset. The results are for practice and are not suitable for medical decisions.

## Results from this run

The holdout set contains 276 records. These scores use the fixed 70/30 split in the notebook and a 0.50 classification threshold.

| Model | ROC-AUC | Average precision | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| RBF SVM | 0.951 | 0.954 | 0.899 | 0.928 | 0.913 | 0.902 |
| Soft Voting | 0.947 | 0.951 | 0.911 | 0.935 | 0.923 | 0.913 |
| Hard Voting | n/a | n/a | 0.904 | 0.922 | 0.913 | 0.902 |

Hard voting produces class labels, so ROC-AUC and average precision are not available for it. In five-fold cross-validation on the training set, soft voting averaged 0.922 ROC-AUC and 0.929 average precision.

## Run the notebook

Download `heart.csv` from the [Heart Failure Prediction dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) and place it at `data/raw/heart.csv`. The data file is not included in this repository.

Use Python 3.10 or newer, then run these commands from the repository folder:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/heart_disease_voting_classifier.ipynb` and run all cells.

## About the data and method

The target column is `HeartDisease` (`1` = positive label, `0` = negative label). The notebook keeps identical rows because the dataset has no patient identifier. Preprocessing is fitted within each training fold, and the test set is held out for final evaluation. The probabilities and scores are not clinically validated.

Dataset: [Federico Sorian, Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).
