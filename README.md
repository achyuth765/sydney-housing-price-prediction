# Sydney Housing Price Prediction

This project develops a machine learning model to predict housing sale prices in three Sydney suburbs:

- Blacktown
- Parramatta
- Randwick

The project includes data preparation, exploratory data analysis, feature engineering, regression modelling, model evaluation, prediction error analysis, and a simple Streamlit web application.

## Project Files

- `Sydney_Housing_Final.ipynb` - Jupyter notebook containing the full machine learning analysis
- `housing_data.csv` - Housing dataset
- `app.py` - Streamlit housing price prediction application
- `housing_price_model.joblib` - Trained machine learning model
- `feature_cols.joblib` - Feature columns used by the model
- `requirements.txt` - Required Python packages

## Run the Application

Install the required packages:

```bash
pip install -r requirements.txt
streamlit run app.py
