import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loading messages and categories from csv filepaths.
    Input:
    messages_filepath: Messages filepath with extension
    categories_filepath: Categories filepath with extension
    Output:
    df: Pandas dataframe containing merged data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df



def clean_data(df):
    """
    Cleaning data in dataframe to tidy format.
    Input:
    df: pandas dataframe containing merged datasets.
    Output:
    df: cleaned data in dataframe.
    """
    categories = df['categories'].str.split(';', expand = True)
    headers = categories.loc[0]

    # Remove the -{number} to get column names
    category_colnames = headers.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])

        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')
        # Extra cleanup
        categories['related'].replace(2, 1, inplace = True)

    df.drop(columns = ['categories'], inplace = True)
    df = pd.concat([df, categories], axis = 1)

    df.drop_duplicates(inplace = True)
    return df

def save_data(df, database_filename):
    """
    Save data from dataframe to database.
    Input:
    df: dataframe containg tidy data
    database_filename: filepath with extension for database
    Output:
    None
    """
    engine = create_engine(('sqlite:///' + database_filename))
    df.to_sql('Messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
