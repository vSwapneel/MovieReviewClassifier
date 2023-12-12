import sys
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
import time
import string
import re
import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize


start_time = time.time()
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

# Regex, used to remove all the characters that are not Alphabets.
pattern = r'[^a-zA-Z\s]'


# Data Preparation for training data
with open('Train Data.txt', 'r', encoding='utf-8') as file:
    _beta_raw_data = file.read()

# Removing the excess white spaces
raw_data = _beta_raw_data.replace("\t", " ").replace("\n", "").replace("<br />", " ")

# Creating a list of reviews (scores included), from the extracted string from the file.
raw_review_entries = raw_data.split("#EOF")


review_data_list =[]
scores_list=[]

# Splitting the scres and reviews from each entry
for sentence in raw_review_entries:
    parts = sentence.split(' ')
    if len(parts) >= 2:
        score = int(parts[0])
        text = ' '.join(parts[1:])
        review_data_list.append(text)
        scores_list.append(score)
        
pre_processed_text = []

# Applying the preprocessing logic on each review of the list.
for entries in review_data_list :
    entry_string =''
    cleaned_text = re.sub(pattern, ' ', entries)
    entry_tokens = word_tokenize(cleaned_text.lower())
    entry_tokens = [word for word in entry_tokens if word not in string.punctuation]
    entry_tokens = [word for word in entry_tokens if word not in stop_words and len(word) > 2]
    entry_tokens = [stemmer.stem(word) for word in entry_tokens]
    entry_string = ' '.join(entry_tokens)
    pre_processed_text.append(entry_string)

    
output_file_path = "Train Output.txt"

# To print the preprocessed result in the output file for validation
with open(output_file_path, 'w') as output_file:
    sys.stdout = output_file
    print(pre_processed_text)
    sys.stdout = sys.__stdout__


tfidf_vectorizer = TfidfVectorizer(max_features=3000, min_df=0.004, max_df=0.6, ngram_range=(1, 3))
X = tfidf_vectorizer.fit_transform(pre_processed_text)


# Data Preparation for testing data
with open('Test Data.txt', 'r', encoding='utf-8') as file:
    _beta_test_data = file.read()

raw_test_data = _beta_test_data.replace("\t", " ").replace("\n", "").replace("<br />", " ")
test_data_entries = raw_test_data.split("#EOF")

test_data_entries.pop()
pre_processed_test_data = []

# Applying the preprocessing logic on each review of the list.
for entries in test_data_entries :
    entry_string =''
    cleaned_text = re.sub(pattern, ' ', entries)
    entry_tokens = word_tokenize(cleaned_text.lower())
    entry_tokens = [word for word in entry_tokens if word not in string.punctuation]
    entry_tokens = [word for word in entry_tokens if word not in stop_words and len(word) > 2]
    entry_tokens = [stemmer.stem(word) for word in entry_tokens]
    entry_string = ' '.join(entry_tokens)
    pre_processed_test_data.append(entry_string)

test_output_file_path = "Test Output.txt"

# To print the preprocessed result in the output file for validation
with open(test_output_file_path, 'w') as output_file:
    sys.stdout = output_file  # Redirect stdout to the file
    print(pre_processed_test_data)  # Print your desired output
    sys.stdout = sys.__stdout__

# Calculated the value of code with cross validation results
k= 631

knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights='distance')


X_test = tfidf_vectorizer.transform(pre_processed_test_data)


knn.fit(X, scores_list)

y_pred = np.sign(np.array(knn.predict(X_test))).astype(int)
np.set_printoptions(threshold=np.inf)

end_time = time.time()
elapsed_time = end_time - start_time

# Gives the total time taken to preprocess, train and fit the results for the classifier
print(f"Elapsed time: {elapsed_time} seconds")

# Print the results in the output file
pred_output_file_path = "Pred Output.dat"
with open(pred_output_file_path, 'w') as pred_output_file_path:
    sys.stdout = pred_output_file_path 
#     To print appropriate results in the output file
    for i, num in enumerate(y_pred):
        if num >= 0:
            formatted_num = "+1"
        else:
            formatted_num = str(num)
        if i != len(y_pred)-1:
            pred_output_file_path.write(formatted_num + "\n")
        else:
            pred_output_file_path.write(formatted_num)
    sys.stdout = sys.__stdout__

# Printint the prediction array in console
print(y_pred)