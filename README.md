# House Price Prediction Using Machine Learning 🏡

An end-to-end Machine Learning regression project that analyzes residential property sales in King County, Washington (Seattle area) and predicts house prices using property characteristics and geographical information.

## 📌 Project Overview

This project explores the factors influencing residential property prices and develops regression models to estimate house sale prices.

The notebook follows a complete data science workflow, including data exploration, data cleaning, feature engineering, visualization, model training, hyperparameter tuning and evaluation.

The target variable is `price` representing the house sale price in US dollars.

## 🎯 Project Objectives

- Explore and understand the King County housing dataset.
- Identify invalid records and data-quality issues.
- Analyze relationships between property features and house prices.
- Perform exploratory data analysis (EDA).
- Engineer meaningful features to improve prediction performance.
- Apply preprocessing techniques and location-based target encoding.
- Train and compare multiple regression algorithms.
- Tune model hyperparameters using cross-validation.
- Evaluate predictions using regression metrics.
- Identify the most influential features affecting house prices.

## 📂 Project Structure

```text
house-price-prediction-ml/
│
├── README.md
├── house_price_analysis.ipynb
├── data.csv
└── requirements.txt
```

## 🛠️ Technologies and Libraries

- **Python** — programming language
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical computing
- **Matplotlib** — data visualization
- **Seaborn** — statistical visualization
- **Scikit-learn** — preprocessing, regression models, cross-validation, hyperparameter tuning, and evaluation
- **Jupyter Notebook / Google Colab** — development environment

## 🔄 Project Workflow

### 1. Data Loading and Exploration

- Load the King County housing dataset.
- Examine dataset dimensions, data types and descriptive statistics.
- Inspect missing values and duplicate records.
- Analyze sale dates, cities, and ZIP codes.
- Investigate the relationship between living area, above-ground area, and basement area.

### 2. Data Cleaning

The notebook identifies and handles several data-quality issues:

- 49 records with zero house prices.
- 6 records with implausible price-per-square-foot values.
- 2 records with zero bedrooms and bathrooms.

A total of 57 invalid records are removed before model training.

The notebook also removes:
- `country` because it contains a single value.
- `street` because it is almost unique for every record.

### 3. Exploratory Data Analysis (EDA)

The analysis investigates:

- House price distributions and skewness.
- Property size and number of bedrooms.
- Bathrooms, floors and property condition.
- Waterfront and renovated properties.
- City-wise and ZIP-code-wise price differences.
- Correlations between numerical features and house prices.
- Relationships between living area and sale price.

Visualizations include histograms, scatterplots, boxplots, correlation heatmaps, and pairwise comparisons.

### 4. Data Preprocessing

The preprocessing pipeline includes:

- Filtering invalid house sales.
- Parsing sale dates.
- Splitting data into training and testing sets (80/20).
- Applying a logarithmic transformation to the target variable `price`.
- Encoding high-cardinality location features.
- Scaling numerical features for the Ridge Regression model.

Target encoding is fitted inside the model pipeline with internal cross-fitting to reduce target leakage.

### 5. Feature Engineering

The project creates additional features to capture property characteristics and relationships:

- `house_age` — age of the property at the time of sale.
- `renovated` — indicates whether a property was renovated.
- `years_since_reno` — years since renovation.
- `has_basement` — indicates basement availability.
- `total_rooms` — combined bedrooms and bathrooms.
- `sqft_per_room` — living area per room.
- `bath_per_bed` — bathroom-to-bedroom ratio.
- `lot_to_living` — relationship between lot size and living area.
- `log_sqft_living` — logarithmic living-area feature.
- `log_sqft_lot` — logarithmic lot-area feature.
- `basement_share` — basement area relative to living area.
- `quality_score` — engineered property quality indicator.
- `is_new` and `is_historic` — property age categories.
- `sale_month` — month of the sale.

Location features include city and ZIP code, which are target-encoded within the model pipeline.

### 6. Data Visualization

The notebook presents visual analysis of:

- House price and log-price distributions.
- Main property feature distributions.
- Price versus living area.
- Price differences across property characteristics.
- City-wise and ZIP-code-wise property prices.
- Feature correlations and pairwise relationships.

### 7. Model Training

Four regression algorithms are compared:

- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- HistGradientBoosting Regressor

Five-fold cross-validation is used to compare model performance using:

- Root Mean Squared Error (RMSE) on log-transformed prices.
- R² score.

GridSearchCV is used for hyperparameter tuning of the regression models.

### 8. Model Evaluation

The tuned models are evaluated on a held-out test dataset.

Evaluation metrics include:

- R² Score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Median Absolute Error
- Mean Absolute Percentage Error (MAPE)

Additional evaluation includes:

- Actual versus predicted price plots.
- Residual analysis.
- Prediction errors across house-price brackets.
- Model comparison visualizations.
- Permutation feature importance.

## 📊 Key Findings and Results

According to the notebook's final evaluation:

- **Best-performing model:** Tuned Gradient Boosting Regressor.
- **R² on log-transformed prices:** Approximately 0.84.
- **R² on actual dollar prices:** Approximately 0.80.
- **Median Absolute Error:** Approximately $50,000.
- **Mean Absolute Error:** Approximately $88,000.
- **MAPE:** Approximately 15.5%.

Prediction accuracy analysis:

- Approximately 47% of predictions fall within ±10% of actual prices.
- Approximately 76% fall within ±20%.
- Approximately 88% fall within ±30%.

### Important Predictive Features

Permutation importance identifies the following influential features:

1. ZIP code
2. Living area (`sqft_living`)
3. City
4. Engineered property quality score
5. Above-ground area
6. Lot size
7. Bathroom-to-bedroom ratio

These results highlight the importance of property size and geographical location in house price prediction.

*Results are based on the notebook's particular dataset, train/test split, and evaluation procedure. They are not guaranteed to generalize to other housing markets or future sale prices.*

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later
- Jupyter Notebook
- Required Python libraries

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/House Price Prediction _ ML.git
   ```

2. Navigate to the project directory:

   ```bash
   cd House Price Prediction _ ML
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook House Price Prediction _ ML.ipynb
   ```

5. Run the notebook cells sequentially.

### Dataset Requirement

Place the dataset in the project directory and name it `data.csv`.

The notebook expects the King County housing dataset with columns such as:

`price`, `date`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `yr_built`, `yr_renovated`, `zipcode`, `city`, `statezip`, and related property features.

Update `DATA_PATH` in the notebook if your dataset is stored elsewhere.

## 💡 Future Improvements

- Incorporate latitude and longitude for more detailed geographical analysis.
- Explore additional property-quality and neighbourhood features.
- Experiment with ensemble learning and stacking.
- Expand hyperparameter search spaces.
- Evaluate performance using additional validation strategies.
- Develop an interactive house price prediction application.
- Investigate model explainability using SHAP.

## 📚 Learning Outcomes

This project demonstrates practical experience with:

- Exploratory Data Analysis
- Data Cleaning and Preprocessing
- Feature Engineering
- Regression Algorithms
- Target Encoding
- Machine Learning Pipelines
- Cross-Validation
- Hyperparameter Optimization
- Regression Model Evaluation
- Residual and Error Analysis
- Permutation Feature Importance

## ⚠️ Disclaimer

This project is intended for educational and portfolio purposes. Predictions are estimates based on historical data and should not be treated as professional property valuations or financial advice.

## 👩‍💻 Author

**Laiba Ashfaq**

Computer Science Graduate | Interested in Artificial Intelligence, Machine Learning, and Data Analysis.

## 📄 License

No license has been specified. Add an appropriate open-source license if you intend to permit reuse or distribution of this project.
