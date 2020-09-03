"""
Date: 10.10.2019
Course: Research Apprenticeship Cognate Identification
Program: The PHILOLOGICON algorithm calculates intra-language distances for each language based on its word list.
The intra-language matrices are then used to calculate KL and Rao distances to obtain inter-language distance matrices.
"""

# import necessary packages and set options for view
import numpy as np
import pandas as pd
import nltk
from math import e
import cmath

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)


# calculate scaled edit distance for each word with each other
def scaled_edit_distance(word1, word2):
    levenshtein_distance = (2 * (nltk.edit_distance(word1, word2))) / (len(word1) + len(word2))
    final_distance = e ** (-levenshtein_distance)
    return final_distance


# calculate normalising constant: sum of all scaled edit distances for each data frame
def normalising_constant(dataframe):
    norm_constant = dataframe.values.sum()
    return norm_constant


# calculate confusion probabilities for each value: scaled edit distance divided by normalizing constant
def conf_prob(distance_value, constant):
    conf_result = distance_value / constant
    return conf_result


# calculate normalizer k(α): Chernoff Coefficient for k(0.5), k(0), K(1)
def chernoff_coeff(array1, array2, k):
    # convert into numbers-array, because only numbers needed
    array1 = array1.values
    array2 = array2.values

    result1 = np.power(array1, (1 - k)) * np.power(array2, k)
    chernoff_result = np.sum(result1)
    return chernoff_result


# calculate first order differential of the normaliser for k’(0.5)
def first_order_diff(array1, array2, k):
    # convert into numbers-array, because only numbers needed
    array1 = array1.values
    array2 = array2.values

    result1 = np.log(np.divide(array2, array1))  # log
    result2 = np.power(array1, (1 - k)) * np.power(array2, k)
    diff_result = np.sum(result1 * result2)
    return diff_result


# calculate second order differential of the normaliser for k’’(0.5)
def second_order_diff(array1, array2, k):
    # convert into numbers-array, because only numbers needed
    array1 = array1.values
    array2 = array2.values

    result1 = np.log2(np.divide(array2, array1))  # log2
    result2 = np.power(array1, (1 - k)) * np.power(array2, k)
    second_diff_result = np.sum(result1 * result2)
    return second_diff_result


# get intersection of Swadesh-list between 2 matrices and calculate Kullback-Leibler distance
def kullback_leibler_distance(matrix1, matrix2):

    # intersection between row and col names of df1 and df2
    intersec = matrix1.index.intersection(matrix2.index)

    # reindex df1: only keep cols and rows based on intersection
    new_df1 = matrix1.reindex(intersec, axis=1).reindex(intersec, axis=0)

    # reindex df2: only keep cols and rows based on intersection
    new_df2 = matrix2.reindex(intersec, axis=1).reindex(intersec, axis=0)

    # calculate KL distance based on first order differential for k(1) and k(0)
    kl_distance_result = (first_order_diff(new_df1, new_df2, 1) - first_order_diff(new_df1, new_df2, 0)) / 2

    return kl_distance_result


# get intersection of Swadesh-list between 2 matrices and calculate Rao distance: based on k(0.5), k’’(0,5) and k’(0.5)
def rao_distance(matrix1, matrix2):
    # intersection between row and col names of df1 and df2
    intersec = matrix1.index.intersection(matrix2.index)

    # reindex df1: only keep cols and rows based on intersection
    new_df1 = matrix1.reindex(intersec, axis=1).reindex(intersec, axis=0)

    # reindex df2: only keep cols and rows based on intersection
    new_df2 = matrix2.reindex(intersec, axis=1).reindex(intersec, axis=0)

    # calculate Rao distance with k(0.5), k’’(0,5) and k’(0.5)
    result1 = (chernoff_coeff(new_df1, new_df2, 0.5) * second_order_diff(new_df1, new_df2, 0.5)) - (
            first_order_diff(new_df1, new_df2, 0.5) ** 2)
    result2 = result1 / (chernoff_coeff(new_df1, new_df2, 0.5) ** 2)
    rao_distance_result = abs(cmath.sqrt(result2))

    return rao_distance_result


# read data from file
original_data = pd.read_csv('ielex.tsv', sep='\t', header=0)

# remove columns not needed for calculation
data = original_data.drop(['notes', 'iso_code', 'global_id', 'local_id', 'cognate_class', 'tokens'], axis=1)

# sort data
sorted_data = data.sort_values(by=['language'])  # sort data by language
sorted_data = sorted_data.drop_duplicates(
    subset=['language', 'gloss'])  # drop duplicate words based on language and gloss
sorted_data = sorted_data.set_index(['language'])  # set language as index

# create list of dfs to use for language-language-distance later on
df_list = []  # to append the confusion matrices
lang_list = []  # to append the languages

# this block of code calculates the intra-language distance for each language
# grouping data according to language: splitting the df for further operations
for (language, group) in sorted_data.groupby('language'):
    # creating a list of words from transcription-column of each language-group
    word_list = [word for word in group['transcription']]

    # create a list of the meanings from gloss-column of each language-group
    gloss_list = [num for num in group['gloss']]

    # for multi-column and multi-index:
    # creating two series from word list: to calculate confusion probabilities later on
    df_rows = pd.Series(word_list)
    df_cols = pd.Series(word_list)

    # creating two series from gloss list: for column and index names later on
    df_rows2 = pd.Series(gloss_list)
    df_cols2 = pd.Series(gloss_list)

    # nested apply: create a df by applying the scaled_edit_distance function to df_cols- and df_rows-series
    df = pd.DataFrame(df_rows.apply(lambda x: df_cols.apply(lambda y: scaled_edit_distance(x, y))))

    # set index and columns of that df AFTER applying function to series
    df_index = df.set_index([df_rows2, df_rows], inplace=True)  # create multi index for calculations
    df.columns = [df_cols2, df_cols]

    # N(ws):calculate the sum of all edit distances applying normalising_constant function
    constant = normalising_constant(df)

    # calculate confusion probabilities: divide each cell value by N(ws)
    # take each cell as argument and apply conf_prob function to it
    df.loc[(df_cols2, df_cols), (df_rows2, df_rows)] = df.loc[(df_cols2, df_cols), (df_rows2, df_rows)].applymap(
        lambda x: conf_prob(x, constant))

    # set index and columns again: remove phonemic transcription index
    # now we are left with only the glosses as columns and indices for each language
    df.index = df_rows2
    df.columns = df_cols2
    # print(language)
    # print(df)

    # append the matrices to the df-list and append languages to lang-list created earlier
    # they are used to calculate Rao and KL distances
    df_list.append(df)
    lang_list.append(language)

# this block of code uses the intra-language distances to calculate Rao and KL distance
# create two lists for results of KL-distance and Rao-distance and use later to fill final dfs
kl_list = []
rao_list = []

# apply Rao distance function to each confusion matrix in the list
# with each other confusion matrix
for df1 in df_list:
    for df2 in df_list:
        rao_list.append(rao_distance(df1, df2))  # append result to list

# reshape list by converting to array and fill final df
# transpose for a better column-wise view
rao_distance_df = pd.DataFrame(np.array(rao_list).reshape(52, 52), columns=lang_list, index=lang_list).transpose()
print(rao_distance_df)

'''
# OPTIONAL: open txt file, fill with the Rao distance matrix result and close
txt_file = open('result_RAO.txt', 'a')
txt_file.write(rao_distance_df.to_string())
txt_file.close()

'''

# apply KL distance function to each confusion matrix in the list
# with each other confusion matrix
for df1 in df_list:
    for df2 in df_list:
        kl_list.append(kullback_leibler_distance(df1, df2))

# reshape list by converting to array and fill df. transpose for a better column-wise view
kl_distance_df = pd.DataFrame(np.array(kl_list).reshape(52, 52), columns=lang_list, index=lang_list).transpose()
print(kl_distance_df)

'''
# OPTIONAL: open txt file, fill with the KL distance matrix result and close
txt_file2 = open('result_KL.txt', 'a')
txt_file2.write(kl_distance_df.to_string())
txt_file2.close()

'''
