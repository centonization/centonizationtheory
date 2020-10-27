from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import random
import seaborn as sn
from sklearn import preprocessing

def get_tfidf_distributions(all_recordings):
    """
    For a list of all recordings patterns tf-df-results
    Returns:
        list, each elecment a recording, summarised as a list of (pattern, tf-idf, total_count_of_pattern)
    """
    token_pattern = '[A-a0-9#-]*'
    vectorizer = TfidfVectorizer(lowercase=False, sublinear_tf=True, token_pattern=token_pattern)
    X = vectorizer.fit_transform(all_recordings)
    top_patterns = []
    sorted_vocab = sorted([(k,v) for k,v in vectorizer.vocabulary_.items()], key=lambda y: y[1])
    vocab = [x[0] for x in sorted_vocab]
    for i,x in enumerate(X):
        weights = x.toarray()[0]
        pattern_counts = Counter(all_recordings[i].split(' '))
        this_patterns = []
        for v,w in zip(vocab, weights):
            tf = float(pattern_counts[v])/len(pattern_counts)
            this_patterns.append((v, tf*w, pattern_counts[v]))
        top_patterns.append(this_patterns)
    return top_patterns


def top_n(distributions, n=5):
    """n=0 to return all"""
    if not n:
        n = len(distributions)
    return sorted(distributions, key=lambda y: y[1], reverse=True)[:n]


def get_nawba_frame(distributions, nawba, nawba_mappings, nawba_indices):
    """
    For distributions from get_tfidf_distributions return dataframe of all weightings across all nawbas
    """
    distributions = {k:v for k,v in zip(nawba_indices, distributions)}[nawba]
    frame = pd.DataFrame(distributions, columns=['pattern', 'tfidf'])
    frame['is_pattern'] = frame['pattern'].apply(lambda y: any([y in p for p in nawba_mappings[nawba]]))
    return frame


def average_tfidf(distributions, indices):
    """
    For a list of tfidf results (from model.get_tfidf_distributions)
    Return:
        Dataframe of average tf-idf across recordings of the same nawba
        ...[nawba, pattern, tf-idf, frequency]
    """
    frame = zip_nawba(distributions, indices)
    frame_grouped = frame.groupby(['index', 'pattern'])\
                         .agg({'tf-idf': 'mean', 'frequency': 'sum'})\
                         .reset_index()
    return frame_grouped


def zip_nawba(distributions, indices):
    """
    Convert distributions output to DF
    """
    zip_nawba = [
        [(n,x,y,z) for x,y,z in d] \
        for n,d in zip(indices, distributions)
    ]
    frame = pd.DataFrame(
        [y for x in zip_nawba for y in x],
        columns=['index', 'pattern', 'tf-idf', 'frequency']
    )
    return frame


def train_classifier(df, nawba_groups, mbid_nawba):
    patterns = [i for s in nawba_groups.values() for i in s]
            
    frame = df[df['pattern'].isin(patterns)]
    data = frame.pivot_table(values='frequency', columns='pattern', index='index')\
                 .reset_index()
    mbid_nawba_dict = {x:y for x,y in mbid_nawba}
    data['index'] = data['index'].apply(lambda y: mbid_nawba_dict[y])
        
    train, test = train_test_split(data, test_size=0.6)
    
    y = train['index']
    X = train[[x for x in patterns if x in train.columns]]
    
    clf = LogisticRegression(
        random_state=42,
        C=0.01,
        solver='liblinear',
        multi_class='ovr'
    ).fit(X, y)
    
    test_preds = clf.predict(test[[x for x in patterns if x in test.columns]])
    y_true = test['index']
    
    #plt.figure(figsize=(10,10))
    #sn.heatmap(confusion_matrix(test_preds, y_true))
    
    return accuracy_score(test_preds, y_true)