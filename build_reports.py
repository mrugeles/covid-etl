import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

confirmed_path = '../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
deaths_path = '../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
recovered_path = '../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

countries = pd.read_csv(confirmed_path)['Country/Region'].unique()


def process_time_series(path):
    dataset = pd.read_csv(path)
    dataset.drop(['Province/State', 'Lat', 'Long'], axis=1, inplace=True)
    dataset = dataset.groupby('Country/Region').sum()
    dataset = dataset.T
    dataset = dataset.reset_index().rename(columns={'index': 'date'})
    return dataset


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def fit_logistic_curve(predictions):
    p0 = [max(predictions['y']), np.median(predictions['x']), 1, min(predictions['y'])]
    popt, pcov = curve_fit(sigmoid, predictions['x'], predictions['y'], p0, method='dogbox')
    return popt, pcov


def to_polinomyal(x):
    polynomial_features = PolynomialFeatures(degree=4)
    x = x[:, np.newaxis]

    return polynomial_features.fit_transform(x)


def build_model(x, y):
    x = to_polinomyal(x)
    y = to_polinomyal(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=300)

    model = LinearRegression()
    return model.fit(X_train, y_train)


def fit_exponential_curve(x, model):
    x_predic = to_polinomyal(x)
    y_predic = model.predict(x_predic)

    predictions = pd.DataFrame(columns=['x', 'y'])
    predictions['x'] = x_predic[:, 1]
    predictions['y'] = y_predic[:, 1]

    return predictions


def get_exponential_segment(predictions):
    steps = predictions['y']
    growth_factor = np.array([x / steps[i - 1] for i, x in enumerate(steps) if i > 0])
    growth_factor = np.concatenate((np.array([0]), growth_factor))
    peaks, _ = find_peaks(growth_factor, height=0)
    return predictions[peaks[-1]:]


def get_category_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    dataset.drop(['Province/State', 'Lat', 'Long'], axis=1, inplace=True)
    dataset = dataset.groupby('Country/Region').sum().rename({'Country/Region': 'date'})
    return dataset.T


def get_country_dataset(dataset, country_name):
    index = dataset.loc[dataset[country_name] > 0].index[0]
    country_df = dataset.loc[index:][[country_name]].reset_index()
    country_df.rename(columns={'index': 'date', country_name: 'y'}, inplace=True)
    country_df['x'] = np.arange(country_df.shape[0])
    country_df = country_df.iloc[:country_df.shape[0] - 5]
    return country_df


def predict_category(dataset_path, country_name):
    dataset = get_category_dataset(dataset_path)
    country_df = get_country_dataset(dataset, country_name)

    # Fit exponential curve
    model = build_model(country_df['x'], country_df['y'])
    exponential_curve_df = fit_exponential_curve(country_df['x'], model)
    exponential_curve_df = get_exponential_segment(exponential_curve_df)

    # Fit logistic curve
    popt, pcov = fit_logistic_curve(exponential_curve_df)
    x = np.arange(300)
    logisic_curve_df = pd.DataFrame({'x': x, 'y': sigmoid(x, *popt)})

    date = country_df.loc[0]['date']
    logisic_curve_df['date'] = pd.date_range(start=date, periods=logisic_curve_df.shape[0])
    country_df['date'] = pd.date_range(start=date, periods=country_df.shape[0])
    return country_df, logisic_curve_df


def build_reports():
    processed_countries = []
    unprocessed_countries = []

    for country_name in countries:
        try:
            confirmed_df, predicted_confirmed_df = predict_category(confirmed_path, country_name)
            confirmed_df['daily_cases'] = confirmed_df['y'].diff()

            deaths_df, predicted_deaths_df = predict_category(deaths_path, country_name)
            deaths_df['daily_cases'] = deaths_df['y'].diff()

            recovered_df, predicted_recovered_df = predict_category(recovered_path, country_name)
            recovered_df['daily_cases'] = recovered_df['y'].diff()

            confirmed_df.to_csv(f'datasets/countries/{country_name}_confirmed.csv', index=False)
            predicted_confirmed_df.to_csv(f'datasets/countries/{country_name}_predicted_confirmed.csv', index=False)

            deaths_df.to_csv(f'datasets/countries/{country_name}_deaths.csv', index=False)
            predicted_deaths_df.to_csv(f'datasets/countries/{country_name}_predicted_deaths.csv', index=False)

            recovered_df.to_csv(f'datasets/countries/{country_name}_recovered.csv', index=False)
            predicted_recovered_df.to_csv(f'datasets/countries/{country_name}_predicted_recovered.csv', index=False)

            processed_countries += [country_name]
        except:
            unprocessed_countries += [country_name]

    pd.DataFrame(processed_countries, columns=['country']).to_csv('processed_countries.csv', index=False)
    pd.DataFrame(unprocessed_countries, columns=['country']).to_csv('unprocessed_countries.csv', index=False)


if __name__ == "__main__":
    build_reports()
