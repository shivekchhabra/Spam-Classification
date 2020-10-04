import pandas as pd
import re
from ricebowl.processing import data_preproc
from ricebowl.modeling import choose_model
from ricebowl import metrics
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


## Overview:
# This code is a simple spam vs non spam classifier using bag of words (count vectorizer)
# ML model used: Multinomial

def read_data(filepath):
    """
    Specific function to read the data delimmited by \t.

    :param filepath: Path of the data
    :return: dataframe object containing msgs and labels
    """
    data = pd.read_csv(filepath, sep='\t', names=['label', 'message'])
    return data


def lammetization(words):
    """
    General function to lemmatize words and removing stop words.

    :param words: List of words to lemmatize
    :return: Final sentence of lemmatized words.
    """
    lm = WordNetLemmatizer()
    final = [lm.lemmatize(x) for x in words if not x in stopwords.words('english')]
    final = ' '.join(final)
    return final


def basic_processing(data):
    """
    Specific function to preprocess the ham-spam dataset

    :param data:
    :return: updated data after processing
    """
    final_df = pd.DataFrame()
    final_df['label'] = data['label']
    refined_msgs = []
    for i in range(data.shape[0]):
        # Remove ? , . etc.
        msg = re.sub('[^a-zA-Z]', ' ', data['message'][i])
        msg = msg.split()
        final = lammetization(msg)
        refined_msgs = refined_msgs + [final]
    final_df['message'] = refined_msgs
    return final_df


def vectorization(messages):
    """
    General function to vectoize a series of data
    :param messages: dataframe series/list to be vectorized
    :return: Vectorized array
    """
    cv = CountVectorizer()
    vectorized_data = cv.fit_transform(messages).toarray()
    return vectorized_data


if __name__ == '__main__':
    data = read_data('./sms_dataset')
    df = basic_processing(data)
    training_data = vectorization(data['message'])
    data, lb = data_preproc.label_encode(data, c1='label')
    label = data['label']
    xtrain, xtest, ytrain, ytest = data_preproc.train_test_split(training_data, label)
    results = choose_model.multinomial_classifier(xtrain, ytrain, xtest)
    print(metrics.classifier_outputs(ytest, results))
