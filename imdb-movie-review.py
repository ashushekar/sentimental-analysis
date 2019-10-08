import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

reviews = pd.read_csv("movie_data.csv", header=None, skiprows=1)

# load_files returns a branch, containing training texts and training labels
# text_train, y_train = reviews_train.data, reviews_train.target
text, y = reviews[0], reviews[1]
print("type of text: {}".format(type(text)))
print("length of text: {}".format(len(text)))
print("text[1]:\n{}".format(text[1]))

# It seems that data consists of some line breaks (<br />). Hence we need to clean the data
text_cleaned = [line.replace("<br />", " ") for line in text]
print("text_cleaned[1]:\n{}".format(text_cleaned[1]))

# Let us check whether data is evenly distributed per class
print("Samples per class: {}".format(np.bincount(y)))

# Divide dataset into train/test set
text_train, text_test, y_train, y_test = train_test_split(text_cleaned, y, stratify=y,
                                                          test_size=.49,
                                                          random_state=42)
print("Number of documents in text_train: {}".format(len(text_train)))
print("Number of documents in text_test: {}".format(len(text_test)))

"""Now let us apply bag-of-words representation technique
1. Tokenization: Split each document into the words that appear in it(called tokens), 
for example by splitting them on whiteplace and punctuations.
2. Vocabulary Building: Collect a vocabulary of all words that appear in any of the 
documents, and number them.
3. Encoding: For each document, count how often each of the words in the vocabulary
appear in this document
"""
# let is use CountVectorizer which consists of the tokenization of the training data
# and building of the vocabulary
vect = CountVectorizer()
vect.fit(text_train)
X_train = vect.transform(text_train)
print("Vocabulary Size: {}".format(len(vect.vocabulary_)))
# print("Vocabulary Content: {}".format(vect.vocabulary_))

# get_feature_map: returns a convenient list where each entry corresponds to one feature.
feature_names = vect.get_feature_names()
print("Number of feature: {}".format(len(feature_names)))
print("First 10 features: {}".format(feature_names[:10]))
# Print every 2000th feature
print("Every 2000th features: {}".format(feature_names[::2000]))

# Now let us apply some cross validation and logistic regression
scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print("Mean cross validation accuracy: {:.2f}".format(np.mean(scores)))

# We know that Logistic regression had regularization parameter C, which can be tuned
# via cross-validation
grid = GridSearchCV(LogisticRegression(), param_grid={'C': [0.001, 0.01, 0.1, 1, 10]}, cv=5)
grid.fit(X_train, y_train)
print("Best Cross Validation Score: {:.2f}".format(grid.best_score_))
print("Best Parameters: {}".format(grid.best_params_))

# Vectorize the test set
X_test = vect.transform(text_test)
print("Vocabulary Size: {}".format(len(vect.vocabulary_)))
print("Test Set Score: {:.2f}".format(grid.score(X_test, y_test)))

"""Now Let us remove stop words and recheck accuracy"""
vect = CountVectorizer(stop_words="english")
vect.fit(text_train)
X_train_stop_words_rm = vect.transform(text_train)
print("Vocabulary Size: {}".format(len(vect.vocabulary_)))
grid = GridSearchCV(LogisticRegression(), param_grid={'C': [0.001, 0.01, 0.1, 1, 10]}, cv=5)
grid.fit(X_train_stop_words_rm, y_train)
print("Best Cross Validation Score after removing stop-words: {:.2f}".format(grid.best_score_))
print("Best Parameters after removing stop-words: {}".format(grid.best_params_))
X_test_stop_words_rm = vect.transform(text_test)
print("Vocabulary Size after removing stop-words: {}".format(len(vect.vocabulary_)))
print("Test Set Score after removing stop-words: {:.2f}".format(grid.score(X_test_stop_words_rm,
                                                                           y_test)))
