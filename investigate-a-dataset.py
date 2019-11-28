#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset (TMDB Movie Data)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction

# This data set contains informationabout 10,000 movies collected fromThe Movie Database (TMDb), including user ratings and revenues. Some notes:
# - Certain columns, like ‘cast’and ‘genres’, contain multiplevalues separated by pipe (|)characters.
# - There are some odd charactersin the ‘cast’ column. Don’t worryabout cleaning them. You canleave them as is.
# - The final two columns endingwith “_adj” show the budget andrevenue of the associated moviein terms of 2010 dollars,accounting for inflation overtime.
# 
# Let's explore it!

# In[147]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### General Properties

# In[148]:


df = pd.read_csv('tmdb-movies.csv')
df.head(5)


# In[149]:


df.shape


# - this dataset contains 21 columns and 10866 records

# In[150]:


df.describe()


# In[151]:


df.info()


# 
# ### Data Cleaning (Replace this with more specific notes!)

# - There are some columns that are not needed to do analysis, these columns can be dropped

# In[152]:


df.drop(['id', 'imdb_id','homepage','tagline','keywords','overview'], axis=1, inplace=True)


# In[153]:


df.info()


# - There are some empty values in the columns __Cast__, __Director__ and __Genres__ Let's analyze them and try to drop them

# In[154]:


df[df['cast'].isnull()].shape


# In[155]:


df[df['director'].isnull()].shape


# In[156]:


df[df['genres'].isnull()].shape


# They are few records, so they can be dropped.

# In[157]:


df.dropna(inplace=True)
df.info()


# - Also in the dataframe description, It can be appreciated the min value for __revenues__ and __budget__ is 0. It's better to get rid of them to avoid affect the analysis later

# In[158]:


df[(df.revenue==0)&(df.budget==0)].shape


# - There are 4.701 wrong records, let's drop them

# In[159]:


df_wrong_rows = df[(df.revenue==0)&(df.budget==0)]
df.drop(df_wrong_rows.index, axis=0, inplace=True)


# In[160]:


df[(df.revenue==0)&(df.budget==0)].shape


# - Change the datatype of __release_year__ to Int, this column will be used for some analysis later

# In[161]:


df['release_year'] = df['release_year'].astype(int)
df.dtypes


# - Certain columns, like __cast__,__genres__,__director__ and __production_companies__, contain multiplevalues separated by pipe (|) characters. Let's try to create an array with those values

# In[162]:


df['genres'] = df.genres.str.split("|")
df['cast'] = df.cast.str.split("|")
df['director'] = df.director.str.split("|")
df['production_companies'] = df.production_companies.str.split("|")


# In[163]:


df.head(5)


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Research Question 1 (Which genres are most popular from year to year?)

# - Create a new DF with the most popular movies selectiong those ones with __vote_average__ more than the mean
# - The __vote_average__ mean is around 6, this can be checked in dataframe description done at the beginning
# - To reduce the volume of columns, it's better to get only those ones to be anlyzed like __genres__ and __release_year__

# In[164]:


df_aux = df[df.vote_average>=df.vote_average.mean()].loc[:,['genres','release_year']]
df_aux.head(5)


# - As the __genres__ column is an array, it's better to get each value separated with its __release_year__
# - In the next cell there are 2 bucles, one to read row by row the dataframe and the second one is to get each item from the genres array and its __release_year__
# - Then those values are inserted into a new dataframe to analyze and plot the data easily, to do this is necessary to apply "unstack" --> https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.unstack.html

# In[165]:


#create a empty dataframe
df_mp = pd.DataFrame(columns=['genres','release_year'])

for index, row in df_aux.iterrows():
    for genres in row['genres']:
        df_mp = df_mp.append([{'genres':genres,'release_year':row['release_year']}], ignore_index=True)


# In[166]:


df_mp.head(5)


# - Let's apply the __unstack()__ to convert the __genres__ as columns to plot them in a better way

# In[167]:


df_mp_unstack = df_mp.groupby(['release_year','genres']).size().unstack()
df_mp_unstack.head()


# - There are a lot __NaN__ values, better replace with __"0"__

# In[168]:


df_mp_unstack.isnull().sum()


# In[169]:


df_mp_unstack.fillna(0, inplace=True)
df_mp_unstack.isnull().sum()


# - Let's try to plot the new dataframe and get a conclusion

# In[170]:


ax = df_mp_unstack.plot(kind='bar', stacked=True, figsize=(20,15))
ax.set_title('MOST POPULAR GENRES FROM YEAR TO YEAR')
ax.set_ylabel('Total of movies')
ax.set_xlabel('Years')


# Using a stacked bar chart to plot the genres is the best way to take a quick look at the most popular genres over the years, but there are 2 values __Drama__ and __TV Movie__ that use almost the same color and this is not very clear to the human eye, for this reason let's do another analysis

# In[171]:


# Adding a new column with the count by "release_year" and "gender"
df_mp['count'] = df_mp.groupby(['release_year','genres']).genres.transform('count')
df_mp.head()


# - Get the max() __count__ by __release_year__ and __genres__ to confirm with the previous chart what is the most popular __genres__ from year to year

# In[172]:


df_mp.loc[df_mp.groupby(['release_year'])['count'].idxmax()]


# Now it's very clear that pink color is associated to __Drama__ movies

# ### Research Question 2  (What kinds of properties are associated with movies that have high revenues?)

# - Select the movies with the __revenues_adj__ higher than the mean
# - Why considering __revenues_adj__ and not __revenues__? Reading the documentation of the dataset explains this:
#     _The final two columns endingwith “_adj” show the budget andrevenue of the associated moviein terms of 2010 dollars,accounting for inflation overtime_. This value is more accurate

# In[173]:


df_rev = df[df.revenue_adj>=df.revenue_adj.mean()].sort_values(by=['revenue_adj'], ascending=False)
df_rev.head()


# #### Directors associated with movies that have high revenues

# In[189]:


#create a empty dataframe
df_dir = pd.DataFrame(columns=['director','revenue_adj'])

for index, row in df_rev.iterrows():
    for director in row['director']:
        df_dir = df_dir.append([{'director':director,'revenue_adj':row['revenue_adj']}], ignore_index=True)


# In[190]:


ax = df_dir.groupby('director').revenue_adj.sum().sort_values(ascending=False).head(50).plot(kind='bar', figsize=(20,10))
ax.set_title('DIRECTOR ASSOCIATED WITH MOVIES THAT HAVE HIGH REVENUES')
ax.set_xlabel('Directors')
ax.set_ylabel('Revenues')


# Using a bar chart is the best way to find high grossing directors, in this case __Steven Spielberg__ is the most grossing director with a huge difference 

# #### Cast associated with movies that have high revenues

# In[191]:


#create a empty dataframe
df_cast = pd.DataFrame(columns=['cast','revenue_adj'])

for index, row in df_rev.iterrows():
    for cast in row['cast']:
        df_cast = df_cast.append([{'cast':cast,'revenue_adj':row['revenue_adj']}], ignore_index=True)


# In[192]:


ax = df_cast.groupby('cast').revenue_adj.sum().sort_values(ascending=False).head(50).plot(kind='bar', figsize=(20,10))
ax.set_title('CAST ASSOCIATED WITH MOVIES THAT HAVE HIGH REVENUES')
ax.set_xlabel('Cast')
ax.set_ylabel('Revenues')


# It seems __Harrison Ford__ is the high grossing actor with a clear difference over the others.

# #### Production Companies associated with movies that have high revenues

# In[193]:


#create a empty dataframe
df_pcomp = pd.DataFrame(columns=['production_companies','revenue_adj'])

for index, row in df_rev.iterrows():
    for pcomp in row['production_companies']:
        df_pcomp = df_pcomp.append([{'production_companies':pcomp,'revenue_adj':row['revenue_adj']}], ignore_index=True)


# In[196]:


ax = df_pcomp.groupby('production_companies').revenue_adj.sum().sort_values(ascending=False).head(50).plot(kind='bar', figsize=(20,10))
ax.set_title('PRODUCTION COMPANIES ASSOCIATED WITH MOVIES THAT HAVE HIGH REVENUES')
ax.set_xlabel('Production Company')
ax.set_ylabel('Revenues')


# This bar chart presents the high grossing production companies and it's very clear there are 4 or 6 companies that produces the movies that have high revenues, like __Warner Bros__, __Univeral Pictures__, __Paramaount Picture__, __Century Fox Film__ and __Walt Disney Pictures__

# #### Correlation between another variables and movies that have high revenues

# To analyze better the correlation between other variables and the revenues, it's better to create a DF with the values to be analyze and create a scatter plot matrix, to check them is a very quick view

# In[180]:


pd.plotting.scatter_matrix(df_rev.loc[:,['revenue_adj', 'budget_adj', 'popularity','vote_count']], figsize=(15,15))


# It's very clear there is a strong positive correlation between these variables

# ### Research Question 3  (What is the top 5 of the best movies ever?)

# To select the top 5 of the best movies ever, I considered the variables __popularity__, __vote_average__ and __vote_average__

# In[181]:


ax = df.groupby('original_title').popularity.sum().sort_values(ascending=False).head(5).plot(kind='bar')
ax.set_title('TOP 5 - MOST POPULAR MOVIES')
ax.set_xlabel('Movie original name')
ax.set_ylabel('Popularity')


# __Jurassic World__ is the most popular movie

# In[182]:


ax = df.groupby('original_title').vote_count.sum().sort_values(ascending=False).head(5).plot(kind='bar')
ax.set_title('TOP 5 - MOST VOTED MOVIES')
ax.set_xlabel('Movie original name')
ax.set_ylabel('Vote count')


# It can be appreciated that __Inception__ is the most voted movie, but the other movies are very near to the first one.

# In[183]:


ax = df.groupby('original_title').vote_average.sum().sort_values(ascending=False).head(5).plot(kind='bar')
ax.set_title('TOP 5 - HIGHER VOTE AVERAGE MOVIES')
ax.set_xlabel('Movie original name')
ax.set_ylabel('Vote average')


# __Hercules__ is the movie with the higher vote average but for the other ones the result is very tight

# <a id='conclusions'></a>
# ## Conclusions
# 
# - It seems the most popular genre over the years is __Drama__
# 
# - It can be apprecited the movies that have high revenues are produced by __Warner Bros__ company or directed by __Steven Spielberg__ or starring __Harrison Ford__ 
# 
# - It seems there is strong positive correlation between the movies that have high revenues and the __budget_adj__, __popularity__ and __vote_count__
# 
# - And for the last, about the top 5 of best movies ever considering this dataset depends of certains variables, for example the movie most popular is __Jurassic World__, the movie most voted is __Inception__ and the movie with the higher vote average is __Hercules__
# 
# - There was some limitations during the analysis, for example not only the __cast__ and __genres__ contains multiple values separated by __"|"__ also the __production_company__ has multiples values. 
# 
# - About movies with empty __revenue__ and __budget__ may cause confusion, so it's better to get rid of them
# 
# - Another limitation was the __release_year__ that was a string and trying to plot the __MOST POPULAR GENRES FROM YEAR TO YEAR__ was not sorted out properly, for this reason the __release_year__ was converted to "Int"
# 
# - For the last, I can consider this dataset as an effective representation of the population because I could draw some samples and obtain some important conclusions
