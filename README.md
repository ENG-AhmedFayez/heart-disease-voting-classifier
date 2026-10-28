# Heart Disease Voting Classifier

A reproducible notebook comparing baseline classifiers with hard and soft voting on the Heart Failure Prediction dataset. The workflow uses stratified sampling, fold-safe preprocessing pipelines, probability-based ROC-AUC, and a held-out test set.

> **Educational use only.** This model is not a medical device, diagnostic tool, or substitute for professional clinical judgment.

## Project contents

- `notebooks/heart_disease_voting_classifier.ipynb` ? exploration, preprocessing, model comparison, voting ensembles, metrics and limitations.
- `data/raw/` ? put the downloaded CSV here; raw data is intentionally excluded from Git.
- `requirements.txt` ? Python dependencies.

## Dataset

Download `heart.csv` from [Heart Failure Prediction on Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) and save it as `data/raw/heart.csv`. Review and follow the dataset's license and terms before using or redistributing it. The data is not included in this repository.

The dataset target is `HeartDisease` (`1` for the positive label, `0` for the negative label). The notebook checks its schema and preserves exact repeated rows because the dataset does not provide a patient identifier.

## Run locally

Python 3.10 or newer is recommended. From the repository root:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/heart_disease_voting_classifier.ipynb` and run all cells. The notebook searches for the dataset in `data/raw/heart.csv` first.

## Example results

Results from the included run using the fixed stratified split (276 holdout records). Values are specific to this split and the default 0.50 threshold.

| Model | ROC-AUC | Average precision | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| RBF SVM | 0.951 | 0.954 | 0.899 | 0.928 | 0.913 | 0.902 |
| Soft Voting | 0.947 | 0.951 | 0.911 | 0.935 | 0.923 | 0.913 |
| Hard Voting | n/a | n/a | 0.904 | 0.922 | 0.913 | 0.902 |

Hard voting returns class labels rather than a probability ranking, so ROC-AUC and average precision do not apply. In five-fold cross-validation on training data, soft voting averaged 0.922 ROC-AUC (SD 0.055), 0.929 average precision (SD 0.060), 0.890 recall (SD 0.034), and 0.876 F1 (SD 0.034).

## Evaluation approach

A stratified 70/30 train-test split is created with a fixed seed. Numeric imputation and scaling, plus categorical imputation and one-hot encoding, are fitted inside each model pipeline. Five-fold cross-validation is run on training rows only. The final holdout metrics include ROC-AUC and average precision from probabilities, alongside precision, recall, F1 and accuracy at the default 0.50 threshold.

This is a small, observational educational dataset. A random holdout split does not establish performance on future or external populations. Model probabilities are not calibrated, and no clinical recommendations should be made from this exercise.

## References

- Dataset: [Federico Sorian, Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).
- Voting Classifier overview: [GeeksforGeeks](https://www.geeksforgeeks.org/ml-voting-classifier-using-sklearn/).
- Probability voting implementation: [mlxtend documentation](http://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/).
- Voting classifiers discussion: [Analytics Vidhya](https://medium.com/analytics-vidhya/voting-classifier-in-machine-learning-9534504eba39).
- The supplied notebook also referenced articles by Towards Data Science and Kaggle notebooks on cross-validation, EDA, imbalance and outlier detection. The topic and links informed this reorganization; the modeling workflow has been rewritten.
