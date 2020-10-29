import os
import numpy as np
import music21 as m21


def mbids_per_tab(andalusian_description):
    """
    Function that creates a dictionary of tubu with the corresponding mbids
    :return: tab_mbid_lookup
    """
    tab_mbid_lookup = {}
    for i, row in andalusian_description.iterrows():
        try:
            tn = row["sections"][0]["tab"]["transliterated_name"]
        except:
            tn = ""

        tab_mbid_lookup[row["mbid"]] = tn

    return tab_mbid_lookup


def is_subset(short, long):
    """
    Function that check if a string is shorter string is a substring of a longer one
    """
    i = long.index(short[0])
    return long[i:i+len(short)] == short


def get_consecutive_patterns(score_offset, sia_patterns, scores):
    """
    Function that filter the sia patterns by getting only the consecutive ones using the score offset.
    """
    sia_patterns_consecutive = {}
    for score in scores:
        with open(os.path.join(score_offset, score + '.txt')) as f:
            all_notes_offsets = f.read()
        all_notes_offsets_list = [int(x) for x in all_notes_offsets.split(' ') if x]
        consecutive_patterns = [x for x in sia_patterns[score] if is_subset([y[0] for y in x], all_notes_offsets_list)]
        sia_patterns_consecutive[score] = consecutive_patterns
    return sia_patterns_consecutive


def sia_to_notename(sia_patterns):
    """
    Function that converts sia format pattern to note name and remove duplicated patterns in a score.
    It only outputs patterns between 3 and 7 notes.
    """
    all_sia_patterns = {}
    for score in sia_patterns:
        all_sia_patterns[score] = []
        for x in sia_patterns[score]:
            pattern = [m21.pitch.Pitch(y[1]).name for y in x]
            if len(pattern) >= 3 and len(pattern) <= 7:
                all_sia_patterns[score].append(''.join(pattern))
    all_sia_patterns = {score: list(set(all_sia_patterns[score])) for score in all_sia_patterns}
    return all_sia_patterns


def compute_patterns_per_tab(sia_patterns, mbid_tab_lookup, centones_tab):
    """
    Function that order the output sia patterns by tab
    """
    sia_patterns_tab = {}
    for score in sia_patterns:
        tab = mbid_tab_lookup[score]
        if tab in centones_tab:
            if tab not in sia_patterns_tab:
                sia_patterns_tab[tab] = [x for x in sia_patterns[score]]
            else:
                for p in sia_patterns[score]:
                    if p not in sia_patterns_tab[tab]:
                        sia_patterns_tab[tab].append(p)

    return sia_patterns_tab


def filter_patterns_by_min_n(scores, mbid_patterns, mbid_tab_lookup, all_patterns, n):
    final_patterns = {}
    for tab in all_patterns:
        if tab not in final_patterns:
            final_patterns[tab] = []
        for pattern in all_patterns[tab]:
            if not all(x == pattern[0] for x in pattern[1:]):  # filter patterns with same note
                occurrences_per_score_per_tab = []
                for score in scores:
                    if mbid_tab_lookup[score] == tab and score in mbid_patterns:
                        occurrences_per_score_per_tab.append(mbid_patterns[score].count(pattern))

                if np.mean(occurrences_per_score_per_tab) >= n:
                    final_patterns[tab].append(pattern)
    return final_patterns


def compute_exact_R_P(final_patterns, centones_tab):
    """
    Function tha computes Recall and Precision with exact matches
    """
    true = 0
    for tab in final_patterns:
        for centon in centones_tab[tab]:
            check = False
            for p in final_patterns[tab]:
                if centon == p:
                    check = True
            if check:
                true += 1
    all_centones = len([x for y in centones_tab.values() for x in y])
    all_ours = len([x for y in final_patterns.values() for x in y])
    overall_recall = true / all_centones
    overall_precision = true / all_ours
    return overall_recall, overall_precision


def compute_superstring_R_P(final_patterns, centones_tab):
    """
    Function that computes Recall and Precision for superstring matches
    """
    true = 0
    for tab in final_patterns:
        for centon in centones_tab[tab]:
            check = False
            for p in final_patterns[tab]:
                if centon in p:
                    check = True
            if check:
                true += 1
    all_centones = len([x for y in centones_tab.values() for x in y])
    all_ours = len([x for y in final_patterns.values() for x in y])
    overall_recall = true / all_centones
    overall_precision = true / all_ours
    return overall_recall, overall_precision


def compute_tab_exact_R_P(final_patterns, centones_tab):
    """
    Function that computes Recall and Pecision for exact matches by tab
    """
    true = 0
    for tab in final_patterns:
        for centon in centones_tab[tab]:
            check = False
            for p in final_patterns[tab]:
                if centon == p:
                    check = True
            if check:
                true += 1
        all_centones = len([x for y in centones_tab[tab] for x in y])
    all_ours = len([x for y in final_patterns.values() for x in y])
    overall_recall = true / all_centones
    overall_precision = true / all_ours
    return overall_recall, overall_precision


def compute_tab_superstring_R_P(final_patterns, centones_tab):
    """
    Function that computes Recall and Precision for superstring matches by tab
    """
    true = 0
    for tab in final_patterns:
        for centon in centones_tab[tab]:
            check = False
            for p in final_patterns[tab]:
                if centon in p:
                    check = True
            if check:
                true += 1
        all_centones = len([x for y in centones_tab[tab] for x in y])
    all_ours = len([x for y in final_patterns.values() for x in y])
    overall_recall = true / all_centones
    overall_precision = true / all_ours
    return overall_recall, overall_precision
