import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def to_polinomyal(x):
    polynomial_features= PolynomialFeatures(degree=3)
    x = x[:, np.newaxis]

    return polynomial_features.fit_transform(x)


def build_model(x, y):
    x = to_polinomyal(x)
    y = to_polinomyal(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=300)

    model = LinearRegression()
    return model.fit(X_train, y_train)


def get_predictions(model, x_predic):
    x_predic = to_polinomyal(x_predic)
    y_predic = model.predict(x_predic)

    predictions = pd.DataFrame(columns=['x', 'y'])
    predictions['x'] = x_predic[:, 1]
    predictions['y'] = y_predic[:, 1]

    return predictions


def get_cases_by_day(df):
    df = df.groupby('date').count()
    df.reset_index(inplace=True)
    df = df[['date', 'cases']]
    return df


def get_confirmed(df):
    df['x'] = np.arange(0, df.shape[0])
    df['y'] = df['cases'].cumsum()
    return df


def get_exponential_predictions(df, prediction_days, train_days):
    train_df = df[:train_days]
    model = build_model(train_df['x'], train_df['y'])
    return get_predictions(model, np.arange(0, prediction_days))


def set_growth_factor(predictions):
    steps = predictions['y']
    growth_factor = np.array([x / steps[i - 1] for i, x in enumerate(steps) if i > 0])
    growth_factor = np.concatenate((np.array([0]), growth_factor))
    predictions['growth_factor'] = growth_factor
    return predictions


def logaritmic_parameters(predictions):
    inflection_row = predictions[predictions['growth_factor'] > 1.02].tail(1)
    inflection_idx = inflection_row.index[0]
    inflection_x = inflection_row['x'].max()
    inflection_y = inflection_row['y'].max()
    L = inflection_y * 2
    return inflection_idx, L


def get_logistic_df(predictions, inflection_idx, L, start_date):
    predictions_inv = predictions[:inflection_idx + 1][::-1].copy()
    predictions_b = predictions_inv.copy()

    predictions_b['x'] = np.arange(inflection_idx, inflection_idx * 2 + 1)
    predictions_b['y'] = predictions_b['y'].apply(lambda y: L - y)

    logistic_df = predictions[:inflection_idx]
    logistic_df = logistic_df.append(predictions_b)

    logistic_df.loc[:, 'date'] = pd.date_range(start=start_date, periods=logistic_df.shape[0])
    return logistic_df


def update_confirmed(covid_df, path, category):
    prediction_days = 300
    train_days = 100

    cases_by_day = get_cases_by_day(covid_df)
    aggregated_df = get_confirmed(cases_by_day)
    predictions = get_exponential_predictions(aggregated_df, prediction_days, train_days)
    predictions = set_growth_factor(predictions)
    inflection_idx, L = logaritmic_parameters(predictions)
    logistic_df = get_logistic_df(predictions, inflection_idx, L, aggregated_df['date'].min())

    cases_by_day.to_csv(f'{path}/cases_by_day_{category}.csv')
    aggregated_df.to_csv(f'{path}/aggregated_{category}.csv')
    logistic_df.to_csv(f'{path}/logistic_{category}.csv')


def update_datasets():
    covid_df = pd.read_csv('datasets/covid_df.csv', dtype=str)

    update_confirmed(covid_df, 'datasets', 'confirmed')
    update_confirmed(covid_df, 'datasets', 'deaths')


if __name__ == "__main__":
    update_datasets()
