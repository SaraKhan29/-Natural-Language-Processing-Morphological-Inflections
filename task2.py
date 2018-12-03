#!/usr/bin/env python
from __future__ import print_function
import sys, codecs, os, string, getopt, re
from functools import wraps

#####################################################################################
# Necessary functions implementations

# function to calculate hamming distance between two string
def hamming(s, t):
    return sum(1 for x, y in zip(s, t) if x != y)


# function to align two string using hamming distance
def halign(s, t):
    slen = len(s)
    tlen = len(t)
    minscore = len(s) + len(t) + 1
    for upad in range(0, len(t) + 1):
        upper = '_' * upad + s + (len(t) - upad) * '_'
        lower = len(s) * '_' + t
        score = hamming(upper, lower)
        if score < minscore:
            bu = upper
            bl = lower
            minscore = score

    for lpad in range(0, len(s) + 1):
        upper = len(t) * '_' + s
        lower = (len(s) - lpad) * '_' + t + '_' * lpad
        score = hamming(upper, lower)
        if score < minscore:
            bu = upper
            bl = lower
            minscore = score

    zipped = zip(bu, bl)
    newin = ''.join(i for i, o in zipped if i != '_' or o != '_')
    newout = ''.join(o for i, o in zipped if i != '_' or o != '_')
    return newin, newout


# Recursive implementation of Levenshtein, with alignments returned.
def levenshtein(s, t, inscost=1.0, delcost=1.0, substcost=1.0):

    @memolrec
    def lrec(spast, tpast, srem, trem, cost):
        if len(srem) == 0:
            return spast + len(trem) * '_', tpast + trem, '', '', cost + len(trem)
        if len(trem) == 0:
            return spast + srem, tpast + len(srem) * '_', '', '', cost + len(srem)

        addcost = 0
        if srem[0] != trem[0]:
            addcost = substcost

        return min((lrec(spast + srem[0], tpast + trem[0], srem[1:], trem[1:], cost + addcost),
                    lrec(spast + '_', tpast + trem[0], srem, trem[1:], cost + inscost),
                    lrec(spast + srem[0], tpast + '_', srem[1:], trem, cost + delcost)),
                   key=lambda x: x[4])

    answer = lrec('', '', s, t, 0)
    return answer[0], answer[1], answer[4]


def memolrec(func):
    """Memoizer for Levenshtein."""
    cache = {}

    @wraps(func)
    def wrap(sp, tp, sr, tr, cost):
        if (sr, tr) not in cache:
            res = func(sp, tp, sr, tr, cost)
            cache[(sr, tr)] = (res[0][len(sp):], res[1][len(tp):], res[4] - cost)
        return sp + cache[(sr, tr)][0], tp + cache[(sr, tr)][1], '', '', cost + cache[(sr, tr)][2]

    return wrap


# Function returns prefix, stem and suffix part of both lemma and infelcted form.
def align_prefix_root_suffix(lemma, form):
    """Break lemma/form into three parts:
    IN:  1 | 2 | 3
    OUT: 4 | 5 | 6
    1/4 are assumed to be prefixes, 2/5 the stem, and 3/6 a suffix.
    1/4 and 3/6 may be empty.
    """

    aligned = levenshtein(lemma, form, substcost=1.1)  # Force preference of 0:x or x:0 by 1.1 cost
    aligned_lemma, aligned_form = aligned[0], aligned[1]
    # leading spaces
    l_space = max(len(aligned_lemma) - len(aligned_lemma.lstrip('_')), len(aligned_form) - len(aligned_form.lstrip('_')))

    # trailing spaces
    t_space = max(len(aligned_lemma[::-1]) - len(aligned_lemma[::1].lstrip('_')), len(aligned_form[::-1]) - len(aligned_form[::1].lstrip('_')))

    return aligned_lemma[0:l_space], aligned_lemma[l_space:len(aligned_lemma) - t_space], aligned_lemma[len(aligned_lemma) - t_space:], aligned_form[0:l_space], aligned_form[l_space:len(aligned_lemma) - t_space], aligned_form[len(aligned_lemma) - t_space:]


# function extract a number of prefix-change and suffix change rules based on lemma+inflected form
# returns set of prefix and suffix rules
def get_prefix_suffix_rules(lemma, form):

    lemma_prefix, lemma_root, lemma_suffix, form_prefix, form_root, form_suffix = align_prefix_root_suffix(lemma, form)

    # for prefix rules
    pre_rules = set()
    if len(lemma_prefix) >= 0 or len(form_prefix) >= 0:
        in_pre = '<' + lemma_prefix
        out_pre = '<' + form_prefix

        for i in range(0, len(form_root)):
            pre_rules.add((in_pre + form_root[:i], out_pre + form_root[:i]))
            pre_rules = {(str.replace(x[0], '_', ''), str.replace(x[1], '_', '')) for x in pre_rules}

    # for suffix rules
    in_suff = lemma_root + lemma_suffix + '>'
    out_suff = form_root + form_suffix + '>'

    suf_rules = set()
    for i in range(min(len(in_suff), len(out_suff))):
        suf_rules.add((in_suff[i:], out_suff[i:]))

    suf_rules = {(str.replace(x[0], '_', ''), str.replace(x[1], '_', '')) for x in suf_rules}

    return pre_rules, suf_rules


# Function applies the longest-matching changing rule on the given input lemma based on given feature
def apply_best_rule(lemma, feat, pre_rules_dict, suf_rules_dict):
    """Applies the longest-matching suffix-changing rule given an input
    form and the MSD. Length ties in suffix rules are broken by frequency.
    For prefix-changing rules, only the most frequent rule is chosen."""

    base = "<" + lemma + ">"

    if feat not in pre_rules_dict and feat not in suf_rules_dict:
        return lemma  # If no information about this inflected form, returns the original lemma form

    # if the inflection description feature is in suffix rules dictionary
    if feat in suf_rules_dict:
        suff_applicable_rules = []
        for k,v in suf_rules_dict[feat].items():
            if k[0] in base:
                suff_applicable_rules.append((k[0], k[1], v)) #finds the applicable rules for that lemma and description

        if suff_applicable_rules:
            # finds the best rule based on longest matching suffix changing rule
            best_rule = max(suff_applicable_rules, key=lambda x: (len(x[0]), x[2], len(x[1])))
            base = str.replace(base, best_rule[0], best_rule[1]) # replaces part of lemma based on best rule

    # if the iflection description feature is in prefix rules dictionary
    if feat in pre_rules_dict:
        pref_applicable_rules = []
        for k,v in pre_rules_dict[feat].items():
            if k[0] in base:
                pref_applicable_rules.append((k[0], k[1], v))   #finds the applicable rules for that lemma and description

        if pref_applicable_rules:
            # finds the best rule based on longest matching suffix changing rule
            best_rule = max(pref_applicable_rules, key=lambda x: (x[2]))
            base = str.replace(base, best_rule[0], best_rule[1])

    base = str.replace(base, '<', '')
    base = str.replace(base, '>', '')
    return base


# function returns total number of mismatch in prefix
def numleadingsyms(s, symbol):
    return len(s) - len(s.lstrip(symbol))


# function returns total number of mismatch in suffix
def numtrailingsyms(s, symbol):
    return len(s) - len(s.rstrip(symbol))

# Necessary Functions implementation Ends.
################################################################################################

# This is the main function.
def main(argv):
    GROUP, LINE, ACCURACY, TRAIN, TEST, OUTPUT = False, False, False, False, False, False

    # this section checks command line parametes
    ########################################################################################
    ag = argv[1:]
    if '-g' in ag:
        GROUP = True
    if '-l' in ag:
        LINE = True

    if '-a' in ag:
        ACCURACY = True

    if '-tr' in ag:
        TRAIN = True
        train_index = ag.index('-tr')

    if '-te' in ag:
        TEST = True
        test_index = ag.index('-te')

    if GROUP:
        print("\nGroup L18: Welsh")
        print("Sara Khan, 2571648")
        print("Hasan Md Tusfiqur Alam, 2571663")
        sys.exit(0)

    elif TEST and TRAIN:
        try:
            train_path = ag[train_index + 1]
            test_path = ag[test_index + 1]
        except:
            print('Invalid Argument')
            sys.exit(2)
    else:
        print('Illegal Input Format')
        sys.exit(2)
    ########################################################################################

    avg = 0.0
    pre_rules_dict, suf_rules_dict = {}, {}  # dictionaries initialization to store feature description and its rules

    # checks the path and opens the training file
    if not os.path.isfile(train_path):
        print('No train file in that path!!!')
        sys.exit(2)
    lines = [line.strip() for line in codecs.open(train_path, "r", encoding="utf-8")]
    train_instance = len(lines)

    # First, test if language is prefix biased or prefix biased
    # If prefix biased, then reverse the string
    # for each of the lines, compute cumulatively the total changes in prefix and suffix
    pref_bias, suf_bias = 0, 0
    for l in lines:
        lemma, form, _ = l.split(u'\t')
        aligned = halign(lemma, form)  # Aligns the lemma and inflected form based on hamming distance
        if ' ' not in aligned[0] and ' ' not in aligned[1] and '-' not in aligned[0] and '-' not in aligned[1]:
            pref_bias += numleadingsyms(aligned[0], '_') + numleadingsyms(aligned[1], '_')
            suf_bias += numtrailingsyms(aligned[0], '_') + numtrailingsyms(aligned[1], '_')

    # Read in lines and extract transformation rules from the lemma and inflected form
    for l in lines:
        lemma, form, feat = l.split(u'\t')
        if pref_bias > suf_bias:
            lemma = lemma[::-1]
            form = form[::-1]

        # gets the prefix and suffix rules for a pair by calling function and stores
        pref_rules, suf_rules = get_prefix_suffix_rules(lemma, form)

        # updating the dictionary of feature descriptions
        if feat not in pre_rules_dict and len(pref_rules) > 0:
            pre_rules_dict[feat] = {}
        if feat not in suf_rules_dict and len(suf_rules) > 0:
            suf_rules_dict[feat] = {}

        for r in pref_rules:
            if (r[0], r[1]) in pre_rules_dict[feat]:
                pre_rules_dict[feat][(r[0], r[1])] = pre_rules_dict[feat][(r[0], r[1])] + 1
            else:
                pre_rules_dict[feat][(r[0], r[1])] = 1

        for r in suf_rules:
            if (r[0], r[1]) in suf_rules_dict[feat]:
                suf_rules_dict[feat][(r[0], r[1])] = suf_rules_dict[feat][(r[0], r[1])] + 1
            else:
                suf_rules_dict[feat][(r[0], r[1])] = 1


    # Updating suffix dictionary with new inflection feature rules for welsh langauge
    try:
        # reading the feature file where all the additional rules are written.
        feature_file_path = 'new_features.txt'
        feature_files = [line.strip() for line in codecs.open(feature_file_path, "r", encoding="utf-8")]
        suf_key = ''
        for l in feature_files:
            if ';' in l:
                suf_key = l
            elif '\t' in l:
                rule_from, rule_to = l.split(u'\t')

                # if the feature rule is not present in the rule dictionary, adding them with frequency value 1
                if suf_key in suf_rules_dict.keys():
                    if (rule_from, rule_to) not in suf_rules_dict[suf_key]:
                        suf_rules_dict[suf_key][(rule_from, rule_to)] = 1
                else:
                    suf_rules_dict[suf_key] = {}
                    suf_rules_dict[suf_key][(rule_from, rule_to)] = 1
            else:
                continue
    except:
        pass

    # Evaluation on Test file
    # checking the path and opening the test file
    if not os.path.isfile(test_path):
        print('No test file in that path!!!')
        sys.exit(2)

    testfile = [line.strip() for line in codecs.open(test_path, "r", encoding="utf-8")]

    test_instance = len(testfile)
    numcorrect = 0
    numguesses = 0
    val = testfile[0].split(u'\t')

    # check whether the test file is three or two column
    if len(val) == 3:
        for l in testfile:
            lemma, correct, feat, = l.split(u'\t')
            # check whether prefix or suffix biased, reverse the string for prefix biased and sending it
            if pref_bias > suf_bias:
                lemma = lemma[::-1]

            # calls the function to get the predicted inflected form
            outform = apply_best_rule(lemma, feat, pre_rules_dict, suf_rules_dict)

            # reverse the output string again for prefix biased
            if pref_bias > suf_bias:
                outform = outform[::-1]

            # checks the accuracy
            if outform == correct:
                numcorrect += 1
            numguesses += 1
            if LINE:
                print(outform) # prints the predicted form in Std. output if -l parameter is passed

        # compute the accuracy if -a parameter is passed
        if ACCURACY:
            avg += (numcorrect / float(numguesses)) * 100
            avg = float("{0:.3f}".format(avg))

            # prints the detail accuracy information
            print('\n')
            print("------------------------------------")
            print('trained on: ', train_path)
            print('- training instances: ', str(train_instance))
            print('tested on: ', test_path)
            print('- testing instances: ', str(numguesses))
            print('- correct instances:', str(numcorrect))
            print("- accuracy", str(avg))
            print("------------------------------------\n")

    # for test file with two column
    elif len(val) == 2:
        for l in testfile:
            lemma, feat, = l.split(u'\t')
            if pref_bias > suf_bias:
                lemma = lemma[::-1]
            # calls the function to get the predicted inflected form
            outform = apply_best_rule(lemma, feat, pre_rules_dict, suf_rules_dict)
            if pref_bias > suf_bias:
                outform = outform[::-1]
            numguesses += 1

            if LINE:
                print(outform) # prints the predicted form in Std. output if -l parameter is passed

        # if -a parameter is passed with two column test file, this message will be displayed
        if ACCURACY:
            print('\n')
            print('Target Form is not provided. We cannot compute the Accuracy')
            print("------------------------------------\n")


# program starts from here
if __name__ == "__main__":
    main(sys.argv)
