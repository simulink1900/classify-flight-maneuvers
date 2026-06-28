# Flight Maneuver Classification

3-class classification of flight maneuvers based on accelerometer time-series data (x, y, z measurements).

## Project Structure

```
classify-flight-maneuvers/
├── notebooks/
│   ├── flight_maneuver_eda.ipynb           # Exploratory data analysis
│   ├── flight_maneuver_training.ipynb      # Main training pipeline
│   └── flight_maneuver_submission.ipynb    # Test submission generation
├── src/
│   └── utils.py                            # Feature extraction utilities
├── data/                                    # Symlinks to training/test data
├── output/                                  # Model artifacts & submissions
└── pyproject.toml                          # uv dependencies
```

## Workflow

### 1. Training Pipeline (`flight_maneuver_training.ipynb`)

**Key features:**
- **Train/Validation Split**: 70/30 on maneuver level with stratification
- **Feature Extraction**: 47 statistical features from time-series data
- **Feature Selection**: SelectKBest (f_classif) → 30 features
- **sklearn Pipelines**: Standardization → Feature Selection → Model
- **Models Compared**:
  - XGBoost
  - LightGBM
  - RandomForest
- **Hyperparameter Tuning**: GridSearchCV (3-fold CV on training set)
- **Validation Evaluation**: Unseen validation set for honest performance assessment
- **Model Selection**: Best model by minimum F1 score
- **Final Retraining**: Full dataset before test submission

**Output:**
- `output/best_model.pkl` - Trained pipeline with best hyperparameters
- `output/training_results.json` - Summary of results and hyperparameters

### 2. Submission Pipeline (`flight_maneuver_submission.ipynb`)

**Steps:**
1. Load best trained model from `output/best_model.pkl`
2. Extract features from test set
3. Generate predictions using best model
4. Save submission to `output/submission.csv`

## Usage

### Setup Environment
```bash
cd /home/syaramionak/Projects/classify-flight-maneuvers
uv sync
```

### Train Models
```bash
uv run jupyter lab notebooks/flight_maneuver_training.ipynb
```

### Generate Test Submission
```bash
uv run jupyter lab notebooks/flight_maneuver_submission.ipynb
```

## Key Metrics

**Evaluation Metric**: Minimum F1 score across 3 classes

**Latest Results** (full dataset):
- Best Model: XGBoost
- Validation Accuracy: 0.999
- Validation Min F1: 0.996
- Best Hyperparameters: k=35 features, learning_rate=0.15, max_depth=5, n_estimators=200

## Feature Engineering

Extracted 47 features per maneuver:

**Basic Statistics** (10 per axis × 3 axes):
- mean, std, min, max, median, q25, q75, range, skew, kurtosis

**Derived Features**:
- Acceleration magnitude (mean, std, max)
- Pairwise correlations (3 features)
- Rate of change/derivatives (9 features)
- Observation count

## Technology Stack

- **Data Processing**: pandas, numpy
- **ML Framework**: scikit-learn (pipelines, feature selection, preprocessing)
- **Models**: XGBoost, LightGBM, scikit-learn RandomForest
- **Hyperparameter Tuning**: GridSearchCV
- **Notebooks**: Jupyter Lab

## Notes

- Model selection is based on validation set performance (held-out from training)
- Feature selection is done within the pipeline to prevent data leakage
- GridSearchCV uses 3-fold CV only on training data
- Final model is retrained on combined train+validation data for maximum performance