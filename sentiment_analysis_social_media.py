import numpy as np
import pandas as pd
#from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Tokenization libraries
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import streamlit as st



def main():
    
    # Check if training on Naive Bayes has been done already!
    if 'training_finished' not in st.session_state:
        st.session_state.training_finished = True
        st.session_state.naive_bayes_params = initial_naive_bayes()
    
    # Load saved params
    test_columns, m, test_csv_file, map_xi_zero_y_zero, map_xi_one_y_zero, map_xi_zero_y_one, map_xi_one_y_one, y_zero_probability_original, y_one_probability_original, test_accuracy, negative_precision, negative_recall, positive_precision, positive_recall = st.session_state.naive_bayes_params

    st.markdown(
    """
    <style>
    .twitter-header {
        font-size: 55px;
        font-weight: bold;
        color: #1DA1F2;
    }
    </style>
    """, unsafe_allow_html=True # For streamlit
    )

    st.markdown(
    """
    <style>
    .model-specs {
        font-size: 25px;
        font-weight: bold;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True # For streamlit
    )

    st.markdown(
    """
    <style>
    .positive {
        font-size: 30px;
        font-weight: bold;
        color: #00ff00;
    }
    </style>
    """, unsafe_allow_html=True # For streamlit
    )

    st.markdown(
    """
    <style>
    .negative {
        font-size: 30px;
        font-weight: bold;
        color: #FF0000;
    }
    </style>
    """, unsafe_allow_html=True # For streamlit
    )


    # Applying the custom CSS class to the header
    st.write('<h1 class="twitter-header">Twitter Mental Health Sentiment Analysis</h1>', unsafe_allow_html=True)
    # Creating model summary
    #model_specs_header = '<h2 class="model-specs">' + "Model Specifications (Naïve Bayes):" + '</h2>'
    #st.write(model_specs_header, unsafe_allow_html=True)
    with st.expander("Model Specifications (Naïve Bayes):"):
        #model_specs = "Model Test Accuracy (Negative Sentiment): " + str(round(test_accuracy*100, 2)) + "%<br>" + "Model Test Precision (Negative Sentiment): " + str(round(negative_precision*100, 2)) + "%<br>" + "Model Test Recall (Negative Sentiment): " + str(round(negative_recall*100, 2)) + "%<br><br>"
        #model_specs += "Model Test Accuracy (Positive Sentiment): " + str(round(test_accuracy*100, 2)) + "%<br>" + "Model Test Precision (Positive Sentiment): " + str(round(positive_precision*100, 2)) + "%<br>" + "Model Test Recall (Positive Sentiment): " + str(round(positive_recall*100, 2)) + "%<br><br>"
        #st.markdown(model_specs, unsafe_allow_html=True)
        
        "Model Test Accuracy: " + str(round(test_accuracy*100, 2))
        # Build df for display
        data_header = {
            "Naïve Bayes (Words Only)": [str(round(negative_precision*100, 2)), str(round(negative_recall*100, 2)), str(round(positive_precision*100, 2)), str(round(positive_recall*100, 2))],
        }
        df = pd.DataFrame(data_header, index=["Model Test Precision (Negative Sentiment)", "Model Test Recall (Negative Sentiment)", "Model Test Precision (Positive Sentiment)", "Model Test Recall (Positive Sentiment)"])
        st.table(df)


    # User input
    input_string = st.text_area(label = "", height=70)
    #input_string = st.text_input(label="")
    
    if st.button("Analyze Tweet!"):
        result, confidence = test_input(test_columns, m, input_string, test_csv_file, map_xi_zero_y_zero, map_xi_one_y_zero, map_xi_zero_y_one, map_xi_one_y_one, y_zero_probability_original, y_one_probability_original)
        result_display = result + ": " + str(round(confidence*100, 2)) + "%"

        if (result == "Positive"):
            result_display_web = '<h3 class="positive">' + result_display + '</h3>'
            st.write(result_display_web, unsafe_allow_html=True)
        elif (result == "Negative"):
            result_display_web = '<h4 class="negative">' + result_display + '</h4>'
            st.write(result_display_web, unsafe_allow_html=True)


    # Continually test user inputs
    #while(True):
        #input_string = str(input("Enter sentiment analysis string (or quit): "))

        #if (input_string == "quit"):
        #    break

        #result, confidence = test_input(input_string, test_csv_file, map_xi_zero_y_zero, map_xi_one_y_zero, map_xi_zero_y_one, map_xi_one_y_one, y_zero_probability_original, y_one_probability_original)
        #print(result, confidence)
        #print()

    


def test_input(test_columns, m, input_string, test_csv_file, map_xi_zero_y_zero, map_xi_one_y_zero, map_xi_zero_y_one, map_xi_one_y_one, y_zero_probability_original, y_one_probability_original):
    # Testing
    #df_test = pd.read_csv(test_csv_file)
    #m = df_test.shape[1] - 1 # Number features, excluding label. Should be same as train!

    # This is testing
    y_zero_probability = np.log(y_zero_probability_original)
    y_one_probability = np.log(y_one_probability_original)

    
    # Tokenizing input string
    tokens = preprocess_string(input_string)


    for i in range(m): # For each feature
        #column_name = df_test.columns[i]
        column_name = test_columns[i]
        
        xi_entry = 0
        if column_name in tokens:
            xi_entry = 1 # Token has a given word naive bayes trained on

        # Adding probabilities with LAPLACE SMOOTHING
        map_xi_zero_y_zero_value = map_xi_zero_y_zero.get(column_name, 0)
        map_xi_one_y_zero_value = map_xi_one_y_zero.get(column_name, 0)
        map_xi_zero_y_one_value = map_xi_zero_y_one.get(column_name, 0)
        map_xi_one_y_one_value = map_xi_one_y_one.get(column_name, 0)

        if (xi_entry == 0):
            y_zero_probability += np.log(((map_xi_zero_y_zero_value + 1) / (map_xi_zero_y_zero_value + map_xi_one_y_zero_value + 2)))
            y_one_probability += np.log(((map_xi_zero_y_one_value + 1) / (map_xi_zero_y_one_value + map_xi_one_y_one_value + 2)))

        elif (xi_entry == 1):
            y_zero_probability += np.log(((map_xi_one_y_zero_value + 1) / (map_xi_zero_y_zero_value + map_xi_one_y_zero_value + 2)))
            y_one_probability += np.log(((map_xi_one_y_one_value + 1) / (map_xi_zero_y_one_value + map_xi_one_y_one_value + 2)))


    # 1 is positive, 0 is negative
    # Evaluating accuracy
    y_zero_probability_normalized = (np.exp(y_zero_probability)) / (np.exp(y_zero_probability) + np.exp(y_one_probability))
    y_one_probability_normalized = (np.exp(y_one_probability)) / (np.exp(y_zero_probability) + np.exp(y_one_probability))

    if (y_zero_probability > y_one_probability):
        return "Negative", y_zero_probability_normalized
    if (y_one_probability > y_zero_probability):
        return "Positive", y_one_probability_normalized


def train_test_tokenize(csv_file_tokenized):
    df = pd.read_csv(csv_file_tokenized)

    features = df.drop('Label', axis=1) # axis=1 is column
    labels = df['Label']  # Label column

    # 20% split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    # Convert to train/test csv files
    df_train.to_csv('review_train.csv', index=False)
    df_test.to_csv('review_test.csv', index=False)


def tokenize_csv(csv_file):
    df = pd.read_csv(csv_file) # Or else too long to train
    #df = df.sample(n=2500, replace=False)

    tokenized_reviews = []
    unique_tokens = set()

    # Preprocess each review, keep track of unique words
    #for review in tqdm(df['review']):
    for review in df['review']:
        tokens = preprocess_string(review)
        tokenized_reviews.append(tokens)

        unique_tokens.update(tokens) # Using a set
    

    # Iterate over each review, keep track of which words contained in which reviews
    unique_tokens = sorted(unique_tokens)
    binary_review_grid =[]

    #for current_tokenized_review in tqdm(tokenized_reviews):
    for current_tokenized_review in tokenized_reviews:
        binary_review_tokens = []

        for token in unique_tokens:
            if token in current_tokenized_review:
                binary_review_tokens.append(1) # Word is contained is current review
            else:
                binary_review_tokens.append(0) # Word is NOT contained is current review
        
        binary_review_grid.append(binary_review_tokens) # Create grid
    

    # Build PandasDataframe
    df_cleaned = pd.DataFrame(binary_review_grid, columns=unique_tokens)
    #df_cleaned['Label'] = ((df['Label'] == 'positive')).astype(int) # Add labels column, define positive as 1
    df_cleaned['Label'] = ((df['Label'])).astype(int) # Add labels column, define positive as 1
    df_cleaned.to_csv('reviews_cleaned.csv', index=False) # Save as csv file

#String cleaning/tokenization inspired by: https://medium.com/@zubairashfaque/sentiment-analysis-with-naive-bayes-algorithm-a31021764fb4
#This was the only source of inspiration throughout the whole code, beyond CS109 content!
def preprocess_string(input_string):
    # Lowercase the input string
    input_string = input_string.lower()
    
    # Remove all punctuation
    input_string = input_string.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(input_string)
    
    # Porter stemmer for word stemming
    stemmer = PorterStemmer()
    # English stopwords (words that have no independent meaning) from NLTK
    stopwords_set = set(stopwords.words("english"))
    
    # Apply stemming, filter out all non-stopwords
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords_set]
    
    # Return tokens from cleaned up string
    return tokens





def train_naive_bayes(train_csv_file):

    # Map each xi
    map_xi_zero_y_zero = {} # Number of times y is zero when xi is zero
    map_xi_one_y_zero = {} # Number of times y is zero when xi is one

    map_xi_zero_y_one = {} # Number of times y is one when xi is zero
    map_xi_one_y_one = {} # Number of times y is one when xi is one


    df_train = pd.read_csv(train_csv_file)
    m = df_train.shape[1] - 1 # Number features, excluding label

    # Laplace smoothing. +2 since each feature can be 0 or 1 (adding error in)
    y_zero_probability_original = ((df_train['Label']==0).sum() + 1)/(df_train.shape[0] + 2) # y=0 / total # rows
    y_one_probability_original = ((df_train['Label']==1).sum() + 1)/(df_train.shape[0] + 2) # y=0 / total # rows


    # This is training
    #for index, row in tqdm(df_train.iterrows()):
    for index, row in df_train.iterrows():
        for i in range(m): # For each feature

            column_name = df_train.columns[i]
            xi_entry = row[column_name]
            y = row[m] # Last column by 0 indexing
        
            if (y == 0):
                if (xi_entry == 0):
                    map_xi_zero_y_zero[column_name] = map_xi_zero_y_zero.get(column_name, 0) + 1 # Sets default to 0
                if (xi_entry == 1):
                    map_xi_one_y_zero[column_name] = map_xi_one_y_zero.get(column_name, 0) + 1 # Sets default to 0
            if (y == 1):
                if (xi_entry == 0):
                    map_xi_zero_y_one[column_name] = map_xi_zero_y_one.get(column_name, 0) + 1 # Sets default to 0
                if (xi_entry == 1):
                    map_xi_one_y_one[column_name] = map_xi_one_y_one.get(column_name, 0) + 1 # Sets default to 0

    return map_xi_zero_y_zero, map_xi_one_y_zero, map_xi_zero_y_one, map_xi_one_y_one, y_zero_probability_original, y_one_probability_original


def test_naive_bayes(test_csv_file, map_xi_zero_y_zero, map_xi_one_y_zero, map_xi_zero_y_one, map_xi_one_y_one, y_zero_probability_original, y_one_probability_original):
    # Testing
    df_test = pd.read_csv(test_csv_file)
    count_total = 0
    count_correct = 0
    m = df_test.shape[1] - 1 # Number features, excluding label. Should be same as train!

    negative_count_total_real = 0
    negative_count_correct = 0
    negative_count_predicted = 0

    positive_count_total_real = 0
    positive_count_correct = 0
    positive_count_predicted = 0


    # This is testing
    #for index, row in tqdm(df_test.iterrows()):
    for index, row in df_test.iterrows():
        
        y_zero_probability = np.log(y_zero_probability_original)
        y_one_probability = np.log(y_one_probability_original)
        y = row[m] # Last column by 0 indexing

        input_string = ""
        for i in range(m): # For each feature
            column_name = df_test.columns[i]
            xi_entry = row[column_name]

            # Adding probabilities with LAPLACE SMOOTHING
            map_xi_zero_y_zero_value = map_xi_zero_y_zero.get(column_name, 0)
            map_xi_one_y_zero_value = map_xi_one_y_zero.get(column_name, 0)
            map_xi_zero_y_one_value = map_xi_zero_y_one.get(column_name, 0)
            map_xi_one_y_one_value = map_xi_one_y_one.get(column_name, 0)

            if (xi_entry == 0):
                y_zero_probability += np.log(((map_xi_zero_y_zero_value + 1) / (map_xi_zero_y_zero_value + map_xi_one_y_zero_value + 2)))
                y_one_probability += np.log(((map_xi_zero_y_one_value + 1) / (map_xi_zero_y_one_value + map_xi_one_y_one_value + 2)))

            elif (xi_entry == 1):
                y_zero_probability += np.log(((map_xi_one_y_zero_value + 1) / (map_xi_zero_y_zero_value + map_xi_one_y_zero_value + 2)))
                y_one_probability += np.log(((map_xi_one_y_one_value + 1) / (map_xi_zero_y_one_value + map_xi_one_y_one_value + 2)))

        # Evaluating accuracy, precision, and recall
        if (y == 1):
            negative_count_total_real += 1
        if (y == 0):
            positive_count_total_real += 1

        count_total += 1
        if (y_zero_probability > y_one_probability):
            positive_count_predicted += 1
            if (y == 0):
                positive_count_correct += 1
                count_correct += 1

        if (y_one_probability > y_zero_probability):
            negative_count_predicted += 1
            if (y == 1):
                negative_count_correct += 1
                count_correct += 1
    

    accuracy = (count_correct / count_total)
    negative_precision = (negative_count_correct) / (negative_count_predicted)
    negative_recall = (negative_count_correct) / (negative_count_total_real)
    positive_precision = (positive_count_correct) / (positive_count_predicted)
    positive_recall = (positive_count_correct) / (positive_count_total_real)

    return accuracy, negative_precision, negative_recall, positive_precision, positive_recall


def initial_naive_bayes():
    # Build and create Naive Bayes model at the start
    #csv_file = './data/reviews.csv'
    #tokenize_csv(csv_file)

    #csv_file_tokenized = './data/reviews_cleaned.csv'
    #train_test_tokenize(csv_file_tokenized)

    train_csv_file = './data/review_train.csv'
    test_csv_file = './data/review_test.csv'

    # For test input runtime optimization
    df_test = pd.read_csv(test_csv_file)
    test_columns = df_test.columns
    m = df_test.shape[1] - 1

    # Testing
    map_xi_zero_y_zero, map_xi_one_y_zero, map_xi_zero_y_one, map_xi_one_y_one, y_zero_probability_original, y_one_probability_original = train_naive_bayes(train_csv_file)
    test_accuracy, negative_precision, negative_recall, positive_precision, positive_recall = test_naive_bayes(test_csv_file, map_xi_zero_y_zero, map_xi_one_y_zero, map_xi_zero_y_one, map_xi_one_y_one, y_zero_probability_original, y_one_probability_original)
    #print("Test accuracy: " + str(test_accuracy), end="\n\n")
    #print("Test precision: " + str(precision), end="\n\n")
    #print("Test recall: " + str(recall), end="\n\n")

    
    return test_columns, m, test_csv_file, map_xi_zero_y_zero, map_xi_one_y_zero, map_xi_zero_y_one, map_xi_one_y_one, y_zero_probability_original, y_one_probability_original, test_accuracy, negative_precision, negative_recall, positive_precision, positive_recall

    

# Start by running main
if __name__ == "__main__":
    main()
