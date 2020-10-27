import re 

def reduce_pattern(x):
    return re.sub('R', '', re.sub('\d', '', x))

#def extract_pattern_grams(notes, min_n=2, max_n=2):
#    """
#    For a list of list of notes, <notes>
#    Extract all possible note-grams up to a maximum length of <n>
#    Converts stream of notes to bag-of-patterns
#    """
#    num_notes = len(notes)
#    comb  = []
#    for i in range(num_notes):
#        # Final n patterns are counted more than once
#        n_ = num_notes - i if max_n > num_notes - i else max_n
#        comb.append([notes[i:i+j] for j in range(2,n_+1)])
#    flat = [i for c in comb for i in c]
#    return ' '.join([''.join(x) for x in flat
#                        if len(x) >= min_n
#                        if x[-1].isdigit()
#                        if 'R' not in [x[0], x[-2]]
#                        if not x[0].isdigit()]
#                     )

def extract_pattern_grams(notes, min_n=2, max_n=2):
    """
    For a list of list of notes, <notes>
    Extract all possible note-grams up to a maximum length of <n>
    Converts stream of notes to bag-of-patterns
    """
    num_notes = len(notes)
    comb  = []
    for i in range(num_notes):
        # Final n patterns are counted more than once
        n_ = num_notes - i if max_n > num_notes - i else max_n
        comb.append([notes[i:i+j] for j in range(2,n_+1)])
    flat = [i for c in comb for i in c]
    return ' '.join([''.join(x) for x in flat if len(x) >= min_n if 'R' not in x])