# -*- coding: utf-8 -*-
# API scrape
from psaw import PushshiftAPI

# Basic libraries
import numpy as np
import pandas as pd

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# Natural Language Processing
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

from selenium import webdriver

def scrape_data(subreddit):
   
    # Instantiate
    api = PushshiftAPI()

    # Create list of scraped data
    scrape_list = list(api.search_submissions(subreddit=subreddit,
                                filter=['title', 'subreddit', 'num_comments', 'author', 'subreddit_subscribers', 'score', 'domain', 'created_utc'],
                                limit=50000))

    #Filter list to only show Subreddit titles and Subreddit category
    clean_scrape_lst = []
    for i in range(len(scrape_list)):
        scrape_dict = {}
        scrape_dict['subreddit'] = scrape_list[i][5]
        scrape_dict['author'] = scrape_list[i][0]
        scrape_dict['domain'] = scrape_list[i][2]
        scrape_dict['title'] = scrape_list[i][7]
        scrape_dict['num_comments'] = scrape_list[i][3]
        scrape_dict['score'] = scrape_list[i][4]
        scrape_dict['timestamp'] = scrape_list[i][1]
        clean_scrape_lst.append(scrape_dict)

    # Show number of subscribers
    print(subreddit, 'subscribers:',scrape_list[1][6])
   
    # Return list of scraped data
    return clean_scrape_lst

# Call function and create DataFrame
df_not_onion = pd.DataFrame(scrape_data('nottheonion'))

# Temporaray DataFrame
df_temp = pd.DataFrame()

# Save data to csv
df_not_onion.to_csv('./not_onion.csv')
# df_not_onion = pd.read_csv('./not_onion.csv')

# Shape of DataFrame
print(f'df_not_onion shape: {df_not_onion.shape}')

# Show head
df_not_onion.head()


# Call function and create DataFrame
df_onion = pd.DataFrame(scrape_data('theonion'))

# Save data to csv
df_onion.to_csv('./the_onion.csv')
# df_onion = pd.read_csv('./onion.csv')

# Shape of DataFrame
print(f'df_onion shape: {df_onion.shape}')

# Show head
df_onion.head()

# If you're running this notebook, you can begin with this cell as the data has already been saved to a csv


# r/TheOnion DataFrame
df_onion = pd.read_csv('./the_onion.csv')

# r/nottheonion DataFrame
df_not_onion = pd.read_csv('./not_onion.csv')

# Show first 5 rows of df_onion
print("Shape:", df_onion.shape)
#df_onion.head()


# Show first 5 rows of df_not_onion
print("Shape:", df_not_onion.shape)
#df_not_onion.head()

def clean_data(dataframe):

    # Drop duplicate rows
    dataframe.drop_duplicates(subset='title', inplace=True)
   
    # Remove punctation
    dataframe['title'] = dataframe['title'].str.replace('[^\w\s]',' ')

    # Remove numbers
    dataframe['title'] = dataframe['title'].str.replace('[^A-Za-z]',' ')

    # Make sure any double-spaces are single
    dataframe['title'] = dataframe['title'].str.replace('  ',' ')
    dataframe['title'] = dataframe['title'].str.replace('  ',' ')

    # Transform all text to lowercase
    dataframe['title'] = dataframe['title'].str.lower()
   
    print("New shape:", dataframe.shape)
    return dataframe.head()

# Call `clean_data(dataframe)` function
clean_data(df_onion)

# Call `clean_data(dataframe)` function
clean_data(df_not_onion)

# Create a DataFrame to check nulls
pd.DataFrame([df_onion.isnull().sum(),df_not_onion.isnull().sum()], index=["TheOnion","notheonion"]).T

# Convert Unix Timestamp to Datetime
df_onion['timestamp'] = pd.to_datetime(df_onion['timestamp'], unit='s')
df_not_onion['timestamp'] = pd.to_datetime(df_not_onion['timestamp'], unit='s')

# Show date-range of posts scraped from r/TheOnion and r/nottheonion
print("TheOnion start date:", df_onion['timestamp'].min())
print("TheOnion end date:", df_onion['timestamp'].max())
print("nottheonion start date:", df_not_onion['timestamp'].min())
print("nottheonion end date:", df_not_onion['timestamp'].max())

def bar_plot(x, y, title, color):    
   
    # Set up barplot
    plt.figure(figsize=(9,5))
    g=sns.barplot(x, y, color = color)    
    ax=g

    # Label the graph
    plt.title(title, fontsize = 15)
    plt.xticks(fontsize = 10)

    # Enable bar values
    # Code modified from http://robertmitchellv.com/blog-bar-chart-annotations-pandas-mpl.html
    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for p in ax.patches:
        totals.append(p.get_width())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+.3, p.get_y()+.38, \
                int(p.get_width()), fontsize=10)
       
# Set x values: # of posts
df_onion_authors = df_onion['author'].value_counts()
df_onion_authors = df_onion_authors[df_onion_authors > 100].sort_values(ascending=False)

# Set y values: Authors
df_onion_authors_index = list(df_onion_authors.index)

# Call function
bar_plot(df_onion_authors.values, df_onion_authors_index, 'Most Active Authors: r/TheOnion', 'r')

# Set x values: # of posts
df_not_onion_authors = df_not_onion['author'].value_counts()
df_not_onion_authors = df_not_onion_authors[df_not_onion_authors > 100].sort_values(ascending=False)

# Set y values: Authors
df_not_onion_authors_index = list(df_not_onion_authors.index)

# Call function
bar_plot(df_not_onion_authors.values, df_not_onion_authors_index, 'Most Active Authors: r/nottheonion','b')

# Set x values: # of posts
df_onion_domain = df_onion['domain'].value_counts()
df_onion_domain = df_onion_domain.sort_values(ascending=False).head(5)

# Set y values: Domains
df_onion_domain_index = list(df_onion_domain.index)

# Call function
bar_plot(df_onion_domain.values, df_onion_domain_index, 'Most Referenced Domains: r/TheOnion','r')

# Set x values: # of posts greater than 100
df_nonion_domain = df_not_onion['domain'].value_counts()
df_nonion_domain = df_nonion_domain.sort_values(ascending=False).head(5)

# Set y values: Names of authors
df_nonion_domain_index = list(df_nonion_domain.index)

# Call function
bar_plot(df_nonion_domain.values, df_nonion_domain_index, 'Most Referenced Domains: r/nottheonion','b')

# Combine df_onion & df_not_onion with only 'subreddit' (target) and 'title' (predictor) columns
df = pd.concat([df_onion[['subreddit', 'title']], df_not_onion[['subreddit', 'title']]], axis=0)

#Reset the index
df = df.reset_index(drop=True)

# Preview head of df to show 'TheOnion' titles appear
df.head(2)
# Preview head of df to show 'nottheonion' titles appear
df.tail(2)

# Replace `TheOnion` with 1, `nottheonion` with 0
df["subreddit"] = df["subreddit"].map({"nottheonion": 0, "TheOnion": 1})

# Print shape of df
print(df.shape)

# Preview head of df to show 1s
df.head(2)
# Preview tail of df to show 0s
df.tail(2)

# Set variables to show TheOnion Titles
mask_on = df['subreddit'] == 1
df_onion_titles = df[mask_on]['title']

# Instantiate a CountVectorizer
cv1 = CountVectorizer(stop_words = 'english')

# Fit and transform the vectorizer on our corpus
onion_cvec = cv1.fit_transform(df_onion_titles)

# Convert onion_cvec into a DataFrame
onion_cvec_df = pd.DataFrame(onion_cvec.toarray(),
                   columns=cv1.get_feature_names())

# Inspect head of Onion Titles cvec
print(onion_cvec_df.shape)

# Set variables to show NotTheOnion Titles
mask_no = df['subreddit'] == 0
df_not_onion_titles = df[mask_no]['title']

# Instantiate a CountVectorizer
cv2 = CountVectorizer(stop_words = 'english')

# Fit and transform the vectorizer on our corpus
not_onion_cvec = cv2.fit_transform(df_not_onion_titles)

# Convert onion_cvec into a DataFrame
not_onion_cvec_df = pd.DataFrame(not_onion_cvec.toarray(),
                   columns=cv2.get_feature_names())

# Inspect head of Not Onion Titles cvec
print(not_onion_cvec_df.shape)

# Set up variables to contain top 5 most used words in Onion
onion_wc = onion_cvec_df.sum(axis = 0)
onion_top_5 = onion_wc.sort_values(ascending=False).head(5)

# Call function
bar_plot(onion_top_5.values, onion_top_5.index, 'Top 5 unigrams on r/TheOnion','r')

# Set up variables to contain top 5 most used words in Onion
nonion_wc = not_onion_cvec_df.sum(axis = 0)
nonion_top_5 = nonion_wc.sort_values(ascending=False).head(5)

# Call function
bar_plot(nonion_top_5.values, nonion_top_5.index, 'Top 5 unigrams on r/nottheonion','b')

# Create list of unique words in top five
not_onion_5_set = set(nonion_top_5.index)
onion_5_set = set(onion_top_5.index)

# Return common words
common_unigrams = onion_5_set.intersection(not_onion_5_set)
common_unigrams

# Set variables to show TheOnion Titles
#mask = df['subreddit'] == 1
#df_onion_titles = df[mask]['title']
#
## Instantiate a CountVectorizer
#cv = CountVectorizer(stop_words = 'english', ngram_range=(2,2))
#
## Fit and transform the vectorizer on our corpus
#onion_cvec = cv.fit_transform(df_onion_titles)
#
## Convert onion_cvec into a DataFrame
#onion_cvec_df = pd.DataFrame(onion_cvec.toarray(),
#                   columns=cv.get_feature_names())
#
## Inspect head of Onion Titles cvec
#print(onion_cvec_df.shape)

# Set variables to show NotTheOnion Titles
#mask = df['subreddit'] == 0
#df_not_onion_titles = df[mask]['title']
#
## Instantiate a CountVectorizer
#cv = CountVectorizer(stop_words = 'english', ngram_range=(2,2))
#
## Fit and transform the vectorizer on our corpus
#not_onion_cvec = cv.fit_transform(df_not_onion_titles)
#
## Convert onion_cvec into a DataFrame
#not_onion_cvec_df = pd.DataFrame(not_onion_cvec.toarray(),
#                   columns=cv.get_feature_names())
#
## Inspect head of Not Onion Titles cvec
#print(not_onion_cvec_df.shape)

#
## Set up variables to contain top 5 most used bigrams in r/TheOnion
#onion_wc = onion_cvec_df.sum(axis = 0)
#onion_top_5 = onion_wc.sort_values(ascending=False).head(5)
#
## Call function
#bar_plot(onion_top_5.values, onion_top_5.index, 'Top 5 bigrams on r/TheOnion','r')
#
## Set up variables to contain top 5 most used bigrams in r/nottheonion
#nonion_wc = not_onion_cvec_df.sum(axis = 0)
#nonion_top_5 = nonion_wc.sort_values(ascending=False).head(5)
#
## Call function
#bar_plot(nonion_top_5.values, nonion_top_5.index, 'Top 5 bigrams on r/nottheonion','b')
#
#not_onion_5_list = set(nonion_top_5.index)
#onion_5_list = set(onion_top_5.index)
#
## Return common words
#common_bigrams = onion_5_list.intersection(not_onion_5_list)
#common_bigrams

#np.intp

# Create lists
custom = stop_words.ENGLISH_STOP_WORDS
custom = list(custom)
common_unigrams = list(common_unigrams)
#common_bigrams = list(common_bigrams)

# Append unigrams to list
for i in common_unigrams:
    custom.append(i)

#   
## Append bigrams to list
#for i in common_bigrams:
#    split_words = i.split(" ")
#    for word in split_words:
#        custom.append(word)

# Baseline score
df['subreddit'].value_counts(normalize=True)

X = df['title']
y = df['subreddit']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=42,
                                                    stratify=y)
pipe = Pipeline([('cvec', CountVectorizer()),    
                 ('lr', LogisticRegression(solver='liblinear'))])

# Tune GridSearchCV
pipe_params = {'cvec__stop_words': [None, 'english' ],
               'cvec__ngram_range': [(1,1), (1,3)],
               'lr__C': [0.01, 1]}

## Tune GridSearchCVwithcustom
#pipe_params = {'cvec__stop_words': [None, 'english', custom ],
#               'cvec__ngram_range': [(1,1), (2,2), (1,3)],
#               'lr__C': [0.01, 1]}

gs = GridSearchCV(pipe, param_grid=pipe_params, cv=3)
gs.fit(X_train, y_train);

print("Best score:", gs.best_score_)
print("Train score", gs.score(X_train, y_train))
print("Test score", gs.score(X_test, y_test))

gs.best_params_

pipe = Pipeline([('tvect', TfidfVectorizer()),    
                 ('lr', LogisticRegression(solver='liblinear'))])

# Tune GridSearchCV
pipe_params = {'tvect__max_df': [.75, .98, 1.0],
               'tvect__min_df': [2, 3, 5],
               'tvect__ngram_range': [(1,1), (1,2), (1,3)],
               'lr__C': [1]}

gs = GridSearchCV(pipe, param_grid=pipe_params, cv=3)
gs.fit(X_train, y_train);
print("Best score:", gs.best_score_)
print("Train score", gs.score(X_train, y_train))
print("Test score", gs.score(X_test, y_test))

gs.best_params_

pipe = Pipeline([('cvec', CountVectorizer()),    
                 ('nb', MultinomialNB())])

# Tune GridSearchCV
pipe_params = {'cvec__ngram_range': [(1,1),(1,3)],
               'nb__alpha': [.36, .6]}

gs = GridSearchCV(pipe, param_grid=pipe_params, cv=3)
gs.fit(X_train, y_train);
print("Best score:", gs.best_score_)
print("Train score", gs.score(X_train, y_train))
print("Test score", gs.score(X_test, y_test))

gs.best_params_

pipe = Pipeline([('tvect', TfidfVectorizer()),    
                 ('nb', MultinomialNB())])

# Tune GridSearchCV
pipe_params = {'tvect__max_df': [.75, .98],
               'tvect__min_df': [4, 5],
               'tvect__ngram_range': [(1,2), (1,3)],
               'nb__alpha': [0.1, 1]}

gs = GridSearchCV(pipe, param_grid=pipe_params, cv=5)
gs.fit(X_train, y_train);
print("Best score:", gs.best_score_)
print("Train score", gs.score(X_train, y_train))
print("Test score", gs.score(X_test, y_test))

gs.best_params_

#Instantiate the classifier and vectorizer
LR = LogisticRegression(solver='liblinear')
TFDF = TfidfVectorizer(ngram_range= (1, 1))

# Fit and transform the vectorizor
TFDF.fit(X_train)

Xtfdf_train = TFDF.transform(X_train)
Xtfdf_test = TFDF.transform(X_test)

# Fit the classifier
LR.fit(Xtfdf_train,y_train)

# Create the predictions for Y training data
preds = LR.predict(Xtfdf_test)

#print(LR.score(Xtfdf_test, y_test))


# proj = pd.read_csv('./Proj.txt', sep="\t")
# proj0 = proj.iloc[:,1]
# proj_test= TFDF.transform(proj0)
# proj1 = LR.predict(proj_test)
# if proj1 == 1:
#     res = "News from credible sites"
# else:
#     res = "Satirical News or Fake News"
    
# print("Output:     " + res)  
# print(res)



pred= pd.read_csv('./predict.csv')
pred0 = pred.iloc[0:1,:]
clean_data(pred0)
pred2 = pred0.iloc[0,:]
pred2_test= TFDF.transform(pred2)
pred3 = LR.predict(pred2_test)
if pred3 == 0:
    res = "News from credible sites"
else:
    res = "Satirical News or Fake News"
    
print("Output:     " + res)
#print(res)

print("To check for yourself enter 0")

inputcase = input()

if inputcase=='0':
        
    driver = webdriver.Firefox()
    driver.get("http://google.com")
    driver.maximize_window()
        
    search = pred2
        
    search_element = driver.find_element_by_name('q')
    search_element.send_keys(search)        
    search_element.submit()

else:
    print("Please enter correct choice")
        
# Create a confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, preds)
cnf_matrix


# name  of classes
class_names=[0,1]

# Set fig and axes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Assign True Neg, False Pos, False Neg, True Pos variables
cnf_matrix = np.array(cnf_matrix).tolist()

tn_fp, fn_tp = cnf_matrix

tn, fp = tn_fp
fn, tp = fn_tp

# Print Scores

print("Accuracy:",round(metrics.accuracy_score(y_test, preds)*100, 2),'%')
# print("Precision:",round(metrics.precision_score(y_test, preds)*100, 2), '%')
# print("Recall:",round(metrics.recall_score(y_test, preds)*100, 2), '%')
# print("Specificity:", round((tn/(tn+fp))*100, 2), '%')
print("Misclassification Rate:", round((fp+fn)/(tn+fp+fn+tn)*100, 2), '%')
# Customize stop_words to include `onion` so that it doesn't appear
# in coefficients

stop_words_onion = stop_words.ENGLISH_STOP_WORDS
stop_words_onion = list(stop_words_onion)
stop_words_onion.append('onion')

#Instantiate the classifier and vectorizer
lr = LogisticRegression(C = 1.0, solver='liblinear')
cvec2 = CountVectorizer(stop_words = stop_words_onion)

# Fit and transform the vectorizor
cvec2.fit(X_train)

Xcvec2_train = cvec2.transform(X_train)
Xcvec2_test = cvec2.transform(X_test)

# Fit the classifier
lr.fit(Xcvec2_train,y_train)

# Create the predictions for Y training data
lr_preds = lr.predict(Xcvec2_test)

print(lr.score(Xcvec2_test, y_test))

# Create list of logistic regression coefficients
lr_coef = np.array(lr.coef_).tolist()
lr_coef = lr_coef[0]

# create dataframe from lasso coef
lr_coef = pd.DataFrame(np.round_(lr_coef, decimals=3),
cvec2.get_feature_names(), columns = ["penalized_regression_coefficients"])

# sort the values from high to low
lr_coef = lr_coef.sort_values(by = 'penalized_regression_coefficients',
ascending = False)

df_head = lr_coef.head(10)
df_tail = lr_coef.tail(10)

# merge back together
df_merged = pd.concat([df_head, df_tail], axis=0)

# plot the sorted dataframe
fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
fig.suptitle('Coefficients!', size=14)
ax = sns.barplot(x = 'penalized_regression_coefficients', y= df_merged.index,
data=df_merged)
ax.set(xlabel='Penalized Regression Coefficients')
plt.tight_layout(pad=3, w_pad=0, h_pad=0);

print("The word that contributes the most positively to being from r/TheOnion is",
      df_merged.index[0], "followed by",
      df_merged.index[1], "and",
      df_merged.index[2],".")

print("-----------------------------------")

print("The word that contributes the most positively to being from r/nottheonion is",
      df_merged.index[-1], "followed by",
      df_merged.index[-2], "and",
      df_merged.index[-3],".")

# Show coefficients that affect r/TheOnion
df_merged_head = df_merged.head(10)
exp = df_merged_head['penalized_regression_coefficients'].apply(lambda x: np.exp(x))
df_merged_head.insert(1, 'exp', exp)
df_merged_head.sort_values('exp', ascending=False)

print("As occurences of", df_merged_head.index[0], "increase by 1 in a title, that title is",
      round(df_merged_head['exp'][0],2), "times as likely to be classified as r/TheOnion.")
# Show coefficients that affect r/nottheonion
df_merged_tail = df_merged.tail(10)
exp = df_merged_tail['penalized_regression_coefficients'].apply(lambda x: np.exp(x * -1))
df_merged_tail.insert(1, 'exp', exp)
df_merged_tail.sort_values('exp', ascending=False)

print("As occurences of", df_merged_tail.index[-1], "increase by 1 in a title, that title is",
      round(df_merged_tail['exp'][-1],2), "times as likely to be classified as r/nottheonion.")