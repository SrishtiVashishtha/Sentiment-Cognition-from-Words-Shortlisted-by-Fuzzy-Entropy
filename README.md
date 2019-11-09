# Sentiment-Cognition-from-Words-Shortlisted-by-Fuzzy-Entropy

**Code for the paper**

[Sentiment Cognition from Words Shortlisted by Fuzzy Entropy.](https://ieeexplore.ieee.org/abstract/document/8815855)

**Description**

In Sentiment Analysis, to highlight the correct words which contribute towards sentiment cognition is very difficult. The use of fuzzy entropy is proposed in our work as an novel step to evaluate sentiment of online movie reviews. This paper performs Sentiment Analysis of movie reviews by shortlisting high sentiment cognition words using **_Fuzzy Entropy_**, **_k-means Clustering_** and sentiment lexicon **_SentiWordNet_**. We have addressed this challenging task of simulating the human cognition of words by developing a model that recognizes sentiment based on fuzzy scores derived from SentiWordNet in an automatic manner. The shortlisted words can be fed as input to any supervised classifier: Deep Neural Networks, Simple Neural Networks, SVM, Naive Bayes, etc. Our fuzzy entropy based approach when trained on **_LSTM_** classifier produces higher accuracy as compared to other state-of-the-art-methods of Sentiment Analysis.

**Dataset**

Movie Review Dataset : [IMDB](https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset). Each movie review has several sentences. The IMDB dataset has a training set of 25,000 labeled instances and a testing set of 25,000 labeled instances; the dataset has positive and negative labels balanced in training and testing set. Another movie review dataset : [polarity dataset v2.0](http://www.cs.cornell.edu/people/pabo/movie-review-data/) by Pang and Lee. It contains 1000 positive and 1000 negative reviews.

**Lexicon**
We have used SentiWordNet lexicon.

**Running the model:**

TrainData_reviews.py : code for creating shortlisted words of train data of IMDB (both positive & negative reviews).

TestData_reviews.py : code for creating shortlisted words of test data of IMDB (both positive & negative reviews).

This data can be fed as input to any supervised classifier. 

**Citation**

If using this code, please cite our work using :

>Vashishtha, Srishti, and Seba Susan. "Sentiment Cognition from Words Shortlisted by Fuzzy Entropy." IEEE Transactions on Cognitive and Developmental Systems (2019).
