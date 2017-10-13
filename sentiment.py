import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import  GaussianNB
from sklearn.naive_bayes import BernoulliNB
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    train_pos_dict = {}
    train_neg_dict = {}
    for text_lines in train_pos:
        word_set = set(text_lines)
        for word in word_set:
            if word not in stopwords:
                train_pos_dict[word] = train_pos_dict.get(word, 0) + 1
    for text_lines in train_neg:
        word_set = set(text_lines)
        for word in word_set:
            if word not in stopwords:
                train_neg_dict[word] = train_neg_dict.get(word, 0) + 1
    pos_words = len(train_pos)
    neg_words = len(train_neg)
    pos_words_final = []
    neg_words_final = []
    for word in train_pos_dict.keys():
        word_count = train_pos_dict.get(word)
        if word_count >= pos_words/100 and word_count/2 >= train_neg_dict.get(word):
            pos_words_final.append(word)
    for word in train_neg_dict.keys():
        word_count = train_neg_dict.get(word)
        if word_count >= neg_words/100 and word_count/2 >= train_pos_dict.get(word):
            neg_words_final.append(word)
    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    def get_vec(data, features):
        vec = []
        for text in data:
            vec.append([1 if i in text else 0 for i in features])
        return vec
    features = pos_words_final
    features.extend(neg_words_final);
    train_pos_vec = get_vec(train_pos, features)
    train_neg_vec = get_vec(train_neg, features)
    test_pos_vec = get_vec(test_pos, features)
    test_neg_vec = get_vec(test_neg, features)
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    #Converting to Labeled Sentence
    def labeledSentence(data, label):
        labeled = []
        for id, line in enumerate(data):
            labeled.append(LabeledSentence(line, [label+'_'+str(id)]))
        return labeled
    labeled_train_pos = labeledSentence(train_pos, "train_pos")
    labeled_train_neg = labeledSentence(train_neg, "train_neg")
    labeled_test_pos = labeledSentence(test_pos, "test_pos")
    labeled_test_neg = labeledSentence(test_neg, "test_neg")

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)
    # Use the docvecs function to extract the feature vectors for the training and test data
    
    # Getting feature vectors
    def get_vec(data, label):
        vec = []
        for i in range(0, len(data)-1):
            vec.append(model.docvecs[label+'_'+str(i)])
        return vec

    train_pos_vec = get_vec(train_pos, "train_pos")
    train_neg_vec = get_vec(train_neg, "train_neg")
    test_pos_vec = get_vec(test_pos, "test_pos")
    test_neg_vec = get_vec(test_neg, "test_neg")

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    train_data = train_pos_vec + train_neg_vec
    nb_model = BernoulliNB(alpha=1.0, binarize=None).fit(train_data, Y)
    lr_model = LogisticRegression().fit(train_data, Y)
    return nb_model, lr_model


def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    train_data = train_pos_vec + train_neg_vec
    nb_model = GaussianNB().fit(train_data, Y)
    lr_model = LogisticRegression().fit(train_data, Y)
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # Predicting positives and negatives
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    predict_pos = model.predict(test_pos_vec)
    predict_neg = model.predict(test_neg_vec)
    # Counting
    for predict in predict_pos:
        if predict == "pos":
            true_pos += 1
        else:
            false_neg += 1
    for predict in predict_neg:
        if predict == "neg":
            true_neg += 1
        else:
            false_pos += 1
    # Accuracy calculcation
    accuracy = (true_pos+true_neg) / float(true_pos+true_neg+false_neg+false_pos)

    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (true_pos, false_neg)
        print "neg\t\t%d\t%d" % (false_pos, true_neg)
    print "accuracy: %f" % (accuracy)

if __name__ == "__main__":
    main()
