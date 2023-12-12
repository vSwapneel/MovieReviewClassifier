import sys
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
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from numba import jit, cuda

def _getCrossvalidationResult(X, scores):
    score_list=[]
    max_accuracy =0
    max_k=7
    for k in range (7, 1000, 3):
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights='distance')
        X_train, X_test_raw, y_train, y_test = train_test_split(X, scores, test_size=0.2, random_state=42)
        knn.fit(X_train, y_train)
        prediction_score = np.sign(np.array(knn.predict(X_test_raw))).astype(int)
        accuracy =  accuracy_score(y_test, prediction_score)
        if(accuracy > max_accuracy):
            max_accuracy = accuracy
            max_k = k
        print("Value of k ",k)
        print("Accuracy ", accuracy)
        print(k,",",accuracy)
        list_inter = [k , accuracy]
        score_list.append(list_inter)
    print("score list",score_list)
    
    test_output_file_path = "K value Output.txt"

    with open(test_output_file_path, 'w') as output_file:
        sys.stdout = output_file  # Redirect stdout to the file
        print(score_list)  # Print your desired output
        sys.stdout = sys.__stdout__
    
    print("Max accuracy score is :", max_accuracy)
    return max_k



stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer('english')
pattern = r'[^a-zA-Z\s]'

with open('Train Data.txt', 'r', encoding='utf-8') as file:
    _beta_raw_data = file.read()

raw_data = _beta_raw_data.replace("\t", " ").replace("\n", "").replace("<br />", " ")
raw_review_entries = raw_data.split("#EOF")

review_data_list =[]
scores_list=[]
for sentence in raw_review_entries:
    parts = sentence.split(' ')
    if len(parts) >= 2:
        score = int(parts[0])
        text = ' '.join(parts[1:])
        review_data_list.append(text)
        scores_list.append(score)
        
pre_processed_text = []
for entries in review_data_list :
    entry_string =''
    cleaned_text = re.sub(pattern, ' ', entries)
    entry_tokens = word_tokenize(cleaned_text.lower())
    entry_tokens = [word for word in entry_tokens if word not in string.punctuation]
    entry_tokens = [word for word in entry_tokens if word not in stop_words and len(word) > 2]
    entry_tokens = [stemmer.stem(word) for word in entry_tokens]
    entry_string = ' '.join(entry_tokens)
    pre_processed_text.append(entry_string)

tfidf_vectorizer = TfidfVectorizer(max_features=3000, min_df=0.004, max_df=0.6, ngram_range=(1, 3))
X = tfidf_vectorizer.fit_transform(pre_processed_text)

k = _getCrossvalidationResult(X,scores_list)

print("Value of k where accuracy is maximum", k)
