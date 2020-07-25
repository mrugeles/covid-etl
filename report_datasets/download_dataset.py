import pandas as pd
from sodapy import Socrata


def get_covid_dataset():
    client = Socrata("www.datos.gov.co", None)
    results = client.get_all("gt2j-8ykr")
    covid_df = pd.DataFrame.from_records(results)
    print(covid_df.tail)
    covid_df.rename(columns={'fecha_de_notificaci_n': 'date', 'id_de_caso': 'cases'}, inplace=True)
    covid_df['date'] = pd.to_datetime(covid_df['date'], format='%Y/%m/%d')
    print(covid_df.tail)
    #covid_df.sort_values(by='date', inplace=True)
    return covid_df


if __name__ == "__main__":
    df = get_covid_dataset()
    df.to_csv('datasets/covid_df.csv', index=False)
