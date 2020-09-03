# Cognate_Identification
This program uses Swadesh lists and the PHILOLOGICON algorithm by Ellison & Kirby (2006) to measure language distances.


## What does the project do?
It calculates the distance between 52 languages using Swadesh lists and the PHIOLOLOGICON algorithm by Ellison & Kirby (2006).
First, intra-language distances are computed. Then, the probability matrices are used to compute inter-language distances.

The PHILOLOGICON algorithm involves several steps:
1) Calculate the scaled edit distances for each word with each other word.
2) Calculate the normalising constant necessary for a probability distribution.
3) Use the scaled edit distances and the normalising constant to calculatethe confusion probabilities.
4) Calculate the Hellinger path/Chernoff coefficient, its first, and its second differential 
   for k’(1), k’(0), k(0.5), k’(0.5),k’’(0.5) using the confusion probability matrices.
5) Calculate the symmetrised Kullback-Leibler distance.
6) Calculate the Rao distance.



## Why is it useful?
The goal of distance-based methods in historical linguistics is to compare languages,
for example through alignment algorithms and find cognates between them. The number or percentage of 
cognates helps to infer which languages could be related to each other. 

Subgrouping these languages and constructing phylogenetic trees can help to represent language relationships. 
Many of the distance-based methods (which rely on inter-language distance) face several problems: 
the comparison of IPA-transcribed meaning lists assumes a common phonetic space. 
There exist, however, language-specific sounds and even differences between the same sound 
depending on the word in that language. Additionally, phonetic environments as well as articulatory 
differences between sounds are ignored. Therefore, the existence of a common phonetic space between hundreds 
of languages seems to be a problematic assumption for inter-language comparison. 

To tackle this problem, Ellison & Kirby’s PHILOLOGICON algorithm is based on the psycholinguistic notion 
of confusion probabilities and lexical neighbourhoods. These notions describe how likely it is to confuse
two words in a language based on their phonological forms.
The authors first compare the intra-language distance to create language-specific profiles to then 
compare the inter-language distance between these probability distributions.  


