#!/usr/bin/env python
# coding: utf-8

import pandas as pd

confirmed_path = 'COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
deaths_path = 'COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
recovered_path = 'COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'


def process_time_series(path):
    dataset = pd.read_csv(path)
    dataset.drop(['Province/State', 'Lat', 'Long'], axis = 1, inplace = True)
    dataset = dataset.groupby('Country/Region').sum()
    dataset = dataset.T
    dataset = dataset.reset_index().rename(columns = {'index':'date'})
    return dataset


if __name__ == "__main__":
    print('Build dataset')
    confirmed_df = process_time_series(confirmed_path)
    deaths_df = process_time_series(deaths_path)
    recovered_df = process_time_series(recovered_path)


    countries = set(confirmed_df.columns.values)
    countries = countries.intersection(
        deaths_df.columns.values, recovered_df.columns.values)

    list_countries = []

    for country in countries:
        df = pd.DataFrame(columns=['date', 'country',
                                'confirmed', 'deaths', 'recovered'])
        df['date'] = confirmed_df['date']
        df['country'] = country
        df['confirmed'] = confirmed_df[country]
        df['deaths'] = deaths_df[country]
        df['recovered'] = recovered_df[country]
        list_countries += [df]

    covid19_countries_df = pd.concat(list_countries)
    covid19_countries_df.to_csv(f'datasets/global-covid19-cases.csv', index=False)

    covid_df = covid19_countries_df.loc[(covid19_countries_df['country'] == 'Colombia') & (covid19_countries_df['confirmed'] != '0')]
    covid_df.to_csv('datasets/covid_df.csv', index=False)
