import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import re

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """
        Load data from SQLlite Database
    Args:
        database_filepath: Path of database of messages and categories
    Returns:
        X: Messages
        Y: Categories one hot encoded
        cateogory_names: category names
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath + '.db')
    df = pd.read_sql_table('FigureEight_data.db',con = engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    """
        Tokenize messages
    Args:
        text: Full message
    Returns:
        final_tokens: tokens of imput text
    """
    #normalization and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    
    ## tokenize text and innitiate lemmatizer
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    
    # iterate through each token
    final_tokens = []
    for token in tokens:
        # Steem, lemmatize and remove leading/ trailing white space
        final_token = PorterStemmer().stem(token)
        final_token = lemmatizer.lemmatize(final_token).strip()
        final_tokens.append(final_token)
    return final_tokens


def build_model():
    """
      NLP pipeline - count words, tf-idf, multiple output classifier
    Returns:
        pipeline
    """
    pipeline = Pipeline([
                    ('vect' , CountVectorizer(tokenizer=tokenize)),
                    ('tfidf' , TfidfTransformer()),
                    ('clf' , MultiOutputClassifier(RandomForestClassifier())),
                        ])

    parameters = {
#                 'vect__ngram_range': [(1, 1), (1, 2)],
#                 'tfidf__use_idf': [True, False]
                 }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """
        Evaluate the model performances: f1-score, precison and recall
    Args:
        model: the model to be evaluated
        X_test: X_test dataframe
        y_test: y_test dataframe
        category_names: category names list defined in load data
    Returns:
        perfomances (DataFrame)
    """
    y_pred = model.predict(X_test)
     
    # create empty dataframe for results with columns Category, f1_score, precision and recall
    Category = []
    F1_score = []
    Precision = []
    Recall = []

    # iterate through y_test columns with target variables (features) for scores of each feature
    i = 0
    for category in y_test.columns:
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test[category], y_pred[:,i], average = 'weighted')
        Category.append(category)
        F1_score.append(f1_score)
        Precision.append(precision)
        Recall.append(recall)
        i += 1
    
    metrics = pd.DataFrame(list(zip(Category, F1_score, Precision, Recall)),
                        columns =['Category','F1_score','Precision','Recall'])

    return print(metrics.mean())


def save_model(model, model_filepath):
    """
        Save Pickle Model
        
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
