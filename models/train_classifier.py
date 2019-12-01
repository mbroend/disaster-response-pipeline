# import libraries
# Download nltk pacakges
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion


def load_data(database_filepath):
    """
    Load the messages from the database.
    Input:
    database_filepath: database filtpath with extension
    Output:
    X: messages from database
    Y: Classes from database
    category_names: The class names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM Messages', engine)
    
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    category_names = Y.columns.tolist()
    return  X, Y, category_names 


def tokenize(text):
    """
    Tokenizes and lemmatizes text:
    Input:
    Text: List of strings or string to be tokenized
    Output:
    clean_tokens: List the text tokenized
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Builds the pipeline for the model
    Input:
    None
    Output:
    model: The ML model
    """
    model = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100)))
                    ])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Print metricts of evaluation of ML model
    Input:
    model: The model
    X_test: Test data
    Y_test: Test labels
    category_names: The name of classes for Y_test
    Output:
    None
    """
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred,columns = Y_test.columns)
    for col in category_names:
        print('-----------------------{}-----------------------'.format(col))
        print(classification_report(Y_test[col],y_pred[col], 
                                    labels = [0,1], 
                                    target_names = ['No','Yes']))


def save_model(model, model_filepath):
    """
    Saves the ML model as pkl file.
    Input:
    model: the ML model
    model_filepath: filepath including extension
    """
    joblib.dump(model, model_filepath,  compress = 3)


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