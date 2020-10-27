from compmusic import dunya
import json
import music21
import os
import pandas as pd


def load_and_parse_centones_mapping(mapping_path):
    """
    Load csv of centone mapping to dict
    """
    centones = pd.read_csv(mapping_path, header=None)
    return {
    'Nawba_{}'.format(x[0]):[y.replace(' ','') for y in list(x[1:]) if str(y) != 'nan'] \
            for i, x in centones.iterrows()
    }


def load_and_parse_nawba_tabs(path):
    """
    Load and parse nawba to tab mapping
    """
    with open(path, 'r') as fp:
        nawba_tabs = json.load(fp)

    tabs_nawba = {}
    for k,v in nawba_tabs.items():
        for t in v:
            tabs_nawba[t] = k
    return tabs_nawba


def download_scores(andalusian_description, dunya_token, target_folder='../data/scores'):
    """
    Given descriptive dataframe of andalusian corpus, download scores to taget folder
    returns: list of score ids downloaded with tab name
    """
    dunya.set_token(dunya_token)
    scores = []
    # Some scores do not have the relevant metadata
    for i, row in andalusian_description.iterrows():
        try:
            tn = row['sections'][0]['tab']['transliterated_name']
        except Exception as e:
            print(e)
            tn = ''
        scores.append([row['mbid'], tn])
        
    bad_mbid = []

    for s in scores: # Download scores from dunya through its mbid
        mbid = s[0]
        # Some scores aren't available
        try:
            score_xml = dunya.docserver.file_for_document(mbid, 'symbtrxml')
            name = mbid + '.xml'
            path = os.path.join(target_folder, name)
            open(path, "wb").write(score_xml)
        except Exception as e:
            print(e)
            bad_mbid.append(mbid)

    scores = [x for x in scores if x[0] not in bad_mbid]
    return scores


def pattern_stream_from_score(path, rest_quarter_length):
    """
    Load a score from <path> and return an ordered list of notes
    R represents a rest greater than or equal to <rest_quarter_length>
    ...rests shorter than <rest_quarter_length> are ignored

    Fails if score contains chords
    """

    s = music21.converter.parse(path)
    p = s.parts[0]

    # These are all the notes of the whole piece, fails for chords
    notes_and_rests = p.flat.notesAndRests.stream()
    notes = []
    for n in notes_and_rests:
        if n.isRest:
            if n.duration.quarterLength >= rest_quarter_length:
                notes.append('R')
        else:
            notes.append(n.name)
    return notes







