This file goes together with the main dataset file: stanfordSentimentTreebank.zip

Here are the files containing all the raw scores and vocab. Some of the vocab may not be used in the final dataset. 
This is due to changes in the data acquisition during our earlier iterations.

A raw score is an integer between [1, 25] with 1 being the most negative and 25 most positive.

rawscores_exp12.txt contains an index followed by 3-6 raw scores (most will only have 3, and only a handful have 4-6). Integers are separated by a comma.

sentlex_exp12.txt contains an index, followed by a comma, and the phrase corresponding to the index.

Please note that some of the symbols, such as commas, have been converted to a HTML tags.

The final processed dataset averaged these responses and mapped them to be between [0,1] and then mapped those to the 5 classes using the following thresholds:
[0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]

For more questions, please see the the main dataset file README.txt. Feel free to ask questions on the website:

http://nlp.stanford.edu/sentiment/

