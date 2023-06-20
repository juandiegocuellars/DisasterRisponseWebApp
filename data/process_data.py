import sys
import pandas as pd
import numpy as np


def load_data(messages_filepath, categories_filepath):
    """
    Load data from csv archive.
    
    Args:
        messages_filepath: filepath of message archive.
        categories_filepath: filepath of categories archive.
        
    Returns:
        df: Dataframe with messages and categories merged
    
    """
    
    #Load messages and categories
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merge both datasets
    df = pd.merge(messages, categories, on = 'id')
    
    return df


def clean_data(df):
    """
    Clean DataFrame values and columns
    
    Args:
        df: DataFrame merged from messages and categories
        
    Returns:
        df: Dataframe with messages and categories merged
    
    """
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    row = categories.iloc[0, :]
    # Extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.split('-')[0])
    
    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(columns ='categories', inplace= True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],join='inner', axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df
    
    
def save_data(df, database_filename):
    """
    Save Cleaned DataFrame 
    
    Args:
        df: DataFrame merged from messages and categories
        database_filename: Name of the database
        
    Returns:
        df: Dataframe with messages and categories merged
    
    """
    #Create a Engine Database in SQLlite
    engine = create_engine('sqlite:///'+ database_filename + '.db')
    
    #Transform Dataframe to Database
    df.to_sql(database_filename, engine, index=False)
    
    return


def main():
    if len(sys.argv) == 4:

        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
 = sys.argv[1:]

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
