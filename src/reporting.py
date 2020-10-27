import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np 
import pandas as pd
import re
from collections import Counter
import extraction
import model

plt.rcParams["font.family"] = "serif"

def get_amins_plot(frame_grouped, nawba, nawba_centones):
    """
    Plot distribution from <frame_grouped> for <nawba> that are in <nawba_centones>[<nawba>]
    """
    relevant_patterns = nawba_centones[nawba]
    this_frame = frame_grouped[(frame_grouped['index'] == nawba) & (frame_grouped['pattern'].isin(relevant_patterns))].sort_values(by='tf-idf', ascending=False)
    plt.xticks(rotation=60)
    plt.title('{}, Amins Centones'.format(nawba.replace('_',' ')))
    plt.ylabel('Average tf-idf')
    plt.xlabel('Centone')
    p = plt.bar(this_frame['pattern'], this_frame['tf-idf'])
    return plt


def string_set(string_list):
    return set(i for i in string_list 
               if not any(i in s for s in string_list if i != s))

def get_patterns(frame_grouped, nawba, scores_in_nawba, min_freq, exclude_monopattern, prefer_superstrings=True):
    # Apply selection rules
    this_nawba = frame_grouped[(frame_grouped['index'] == nawba)]
    this_nawba = this_nawba[this_nawba['frequency']/scores_in_nawba > min_freq]
    if exclude_monopattern:
        this_nawba = this_nawba.loc[~this_nawba['pattern'].apply(lambda y: len(set(extraction.reduce_pattern(y))) == 1)]
    
    this_frame = this_nawba.sort_values(by='tf-idf', ascending=False)

    top_patterns = sorted(this_frame['pattern'])
    # If substrings filter to include the largest
    if prefer_superstrings:
        top_patterns_filt = string_set(top_patterns)
    else:
        top_patterns_filt = set(top_patterns)

    this_frame = this_frame[this_frame['pattern'].isin(top_patterns_filt)]

    return this_frame


def get_top_centones_plot(frame_grouped, nawba, nawba_centones, scores_in_nawba, height=12, width=10, min_freq=10, exclude_monopattern=False, prefer_superstrings=True):
    """
    Plot top <n> centones for <nawba> in <frame_grouped>

    Bars marked green are centones in <nawba_centones>[<nawba>]
    Bars marked red are centones that are superstrings of <nawba_centones>[<nawba>]
    Bars marked blue are centones not specified in lookup tables
    
    <scores_in_nawba> is used to normalise pattern frequency
        (number of scores for this Nawba)
    """ 

    nawba_string = nawba.replace('_',' ')

    # Colour scheme
    bar_edge_colour = '#383838'
    canvas_colour = '#ffffff'
    gridline_colour = '#bcbcbc'

    standard_bar_colour = '#f7f7f7'
    amin_bar_colour = '#303030'
    super_amin_bar_colour = '#aaaaaa'

    this_frame = get_patterns(frame_grouped, nawba, scores_in_nawba, min_freq, exclude_monopattern, prefer_superstrings)

    patterns = this_frame['pattern']
    tfidf = this_frame['tf-idf']
    frequencies = [int(x/scores_in_nawba) for x in this_frame['frequency']]
    max_tfidf = max(tfidf)

    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)

    # gridlines beneath other elements
    #ax.xaxis.grid(True, color=gridline_colour, linestyle='dashed', alpha=0.8)
    #ax.set_axisbelow(True)

    # Canvas colour
    ax.set_facecolor(canvas_colour)

    # Custom Legend
    custom_boxes = [Line2D([0], [0], color=standard_bar_colour, lw=4),
                    Line2D([0], [0], color=amin_bar_colour, lw=4),
                    Line2D([0], [0], color=super_amin_bar_colour, lw=4),
                    Line2D([0], [0], color='#000000', lw=0)]
    ax.legend(
        custom_boxes, 
        ["New pattern", "Chaachoo's Pattern", "Superstring of Chaachoo's pattern", "Average frequency per recording in nawba"]
    )

    # Example data
    y_pos = np.arange(len(patterns))

    # Create horizontal bars
    bars = ax.barh(y_pos, tfidf, align='center', color=standard_bar_colour, edgecolor=bar_edge_colour)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(patterns, fontsize=12)
    ax.invert_yaxis()  # labels read top-to-bottom

    # Labels
    plt.title('{}, Highest Ranking Patterns'.format(nawba_string), size=18)
    plt.xlabel('Average tf-idf', size=16)
    plt.ylabel('Pattern', size=16)

    # Annotate with frequency
    for i, t in enumerate(tfidf):
        ax.text(t + max_tfidf/400, i + .25, str(frequencies[i]), color='black')
    
    # Colour bars as per amins centones
    for i, pat in enumerate(patterns):
        if any([x == extraction.reduce_pattern(pat) for x in nawba_centones[nawba]]):
            bars[i].set_color(amin_bar_colour)
            bars[i].set_edgecolor(bar_edge_colour)
            #print('\\textbf{'+pat+'},', end=" ")
        elif any([x in extraction.reduce_pattern(pat) for x in nawba_centones[nawba]]):
            bars[i].set_color(super_amin_bar_colour)
            bars[i].set_edgecolor(bar_edge_colour)
            #print('\\textit{'+pat+'},', end=" ")

        else:
            pass
            #print(pat+',', end=" ")
    #print(len(patterns))
    return patterns


def get_all_patterns(set_tabs, frame_grouped, tab_num_scores, min_freq, exclude_monopattern=False, prefer_superstrings=False, tfidf_thresh=0, return_confidence=False):
    max_tfidf = max(frame_grouped['tf-idf'])
    lim = tfidf_thresh*max_tfidf

    frame_grouped = frame_grouped[frame_grouped['tf-idf']>lim]
    all_patterns = pd.DataFrame(columns=['index','pattern','tf-idf','frequency','num_scores'])
    for t in set_tabs:
        this_df = get_patterns(frame_grouped, t, tab_num_scores[t], min_freq=min_freq, exclude_monopattern=exclude_monopattern, prefer_superstrings=prefer_superstrings)
        all_patterns = all_patterns.append(this_df)
    
    our_patterns = {}
    for t,df in all_patterns.groupby('index'):
        if return_confidence:
            our_patterns[t] = list(zip(df['pattern'].values, df['tf-idf'].values))
        else:
            our_patterns[t] = df['pattern'].values
    return our_patterns


def get_recalls(our_patterns, centones_tab, set_tabs, print_screen=True, match_superstrings=False):
    all_ours = []
    all_his = []
    tot_n = 0

    results_dict = {}
    if print_screen:
        print('Recall Scores')
        print('-------------')
    for t in set_tabs:
        if t in our_patterns:
            ours = our_patterns[t]
        else:
            ours = []
        
        his = centones_tab[t]    
        
        all_ours += list(ours)
        all_his += list(his)
        
        n_ = [is_match(x, set(his), match_superstrings) for x in set(ours)]
        n = len(set([x for y in n_ for x in y if x]))
        tot_n += n
        
        h = len(set(his))
        R = n/h
        P = n/len(ours) if len(ours) > 0 else np.nan
            
        results_dict[t] = (R, P)
        if print_screen:
            print('{t}: {R} ({n}/{h})'.format(R=R, t=t, n=n, h=h))
            #print(P)
            
    h = len(all_his)
    R = tot_n/h
    P = tot_n/len(all_ours) if len(all_ours) > 0 else np.nan

    if print_screen:
        print('\nOverall: {R} ({n}/{h})'.format(R=R, t=t, n=tot_n, h=h))

    results_dict['overall'] = (R,P)
    
    return results_dict


def get_bop(patterns, min_n=3, max_n=10):
    notes = [[y[0] for y in x['notes']] for x in patterns]
    extracted = [extraction.extract_pattern_grams(nt, min_n=3, max_n=10) for nt in notes]
    return extracted


def complete_pipeline(patterns, tab_num_scores, centones_tab, min_n, max_n, min_freq, exclude_monopatterns, prefer_superstrings, match_superstrings=False, tfidf_thresh=0):
    
    tabs = [p['tab'] for p in patterns]
    set_tabs = set(tabs)
    
    # Get Bag of Patterns
    extracted = get_bop(patterns, min_n=min_n, max_n=max_n)
    
    # TF-IDF
    distributions = model.get_tfidf_distributions(extracted)
    
    # Average
    frame_grouped = average_tfidf(distributions, tabs)
    
    # Get Patterns
    our_patterns = get_all_patterns(set_tabs, frame_grouped, tab_num_scores, min_freq=min_freq, exclude_monopattern=True, prefer_superstrings=False, tfidf_thresh=tfidf_thresh)
    
    # Evaluate
    ct = {k:v for k,v in centones_tab.items() if k in set_tabs}
    set_tabs = {x for x in set_tabs if x in set_tabs}
    results_dict = get_recalls(our_patterns, ct, set_tabs, print_screen=False, match_superstrings=match_superstrings)
    recall, precision = results_dict['overall']

    return recall, precision, [len(x) for x in our_patterns.values()]


def average_tfidf(distributions, tabs):
    set_tabs = list(set(tabs))
    tab_num_scores = Counter(tabs)
    index_tab = dict(enumerate(set_tabs))
    tab_index = {v:k for k,v in index_tab.items()}
    all_tabs_index = [tab_index[t] for t in tabs]
    frame_grouped = model.average_tfidf(distributions, tabs)
    return frame_grouped

def is_match(x, st, superstrings=False):
    if not superstrings:
        return [x] if x in st else []
    else:
        return [s for s in st if s in x]
