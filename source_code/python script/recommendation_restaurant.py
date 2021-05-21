# -*- coding: utf-8 -*-
"""
Created on Mon May  3 18:13:24 2021

@author: nishant
"""

import pymongo

import matplotlib as plt      
import pandas as pd
import json
import seaborn as sns
import numpy as np

#connecting to mongodb atlas
client = pymongo.MongoClient('mongodb+srv://bunny:bunny@cluster0.ywhtn.mongodb.net/myFirstDatabase?retryWrites=true&w=majority')
                             


db = client.get_database('myFirstDatabase')

records = db.reviews

records.count_documents({})


list(records.find({'rating': 2}))


df_all = pd.DataFrame(records.find({}))
df = df_all[df_all.groupby(['restaurantId','userid'])['createdAt'].transform('max') == df_all['createdAt']]

df.describe(include = 'all').transpose()

sns.countplot(df['rating'])


#How many users have rated more than n places ?
n = 1
most_rated_users = df['userid'].value_counts()
user_counts = most_rated_users[most_rated_users >= n]
len(user_counts)
user_counts


most_rated_restaurants = df['restaurantId'].value_counts()
most_rated_restaurants


data_final = df[df['userid'].isin(user_counts.index)]
data_final

final_ratings_matrix = df.reset_index().pivot(index = 'userid',
                                        columns = 'restaurantId', values = 'rating').fillna(0)
final_ratings_matrix.head()

df.set_index(['userid', 'restaurantId'], append=True)

#calc the density of matrix
given_num_of_ratings = np.count_nonzero(final_ratings_matrix)
print('given_num_of_ratings: ', given_num_of_ratings)

#Total no. of ratings that could have been given 
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
print('possible_num_of_ratings: ', possible_num_of_ratings)

#Calculate matrix density
density = (given_num_of_ratings / possible_num_of_ratings) * 100
print('density: {:4.2f}%'.format(density))

#collab filter

pivot_data = data_final.pivot(index = 'userid', columns = 'restaurantId', values = 'rating').fillna(0)
pivot_data.shape
pivot_data.head()

pivot_data['user_index'] = np.arange(0, pivot_data.shape[0],1)
pivot_data.head()
pivot_data.set_index(['user_index'], inplace = True)
pivot_data.head()

final_ratings_matrix

from scipy.sparse.linalg import svds

#SVD
U,s, VT = svds(pivot_data, k = 1)

#Construct diagonal array in SVD
sigma = np.diag(s)

#Applying SVD would output 3 parameters namely
print("U = ",U) #Orthogonal matrix
print('************************************************')
print("S = ",s) #Singular values
print('************************************************')
print("VT = ", VT) #Transpose of Orthogonal matrix




all_user_predicted_ratings = np.dot(np.dot(U,sigma), VT)

#Predicted ratings
pred_data = pd.DataFrame(all_user_predicted_ratings, columns = pivot_data.columns)
pred_data.head()


def recommend_places(userID, pivot_data, pred_data, num_recommendations):
    user_index  = userID-1 #index starts at 0

    sorted_user_ratings = pivot_data.iloc[user_index].sort_values(ascending = False) #sort user ratings

    sorted_user_predictions = pred_data.iloc[user_index].sort_values(ascending = False)#sorted_user_predictions
    
    temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis = 1)
    temp.index.name = 'Recommended Places'
    temp.columns = ['user_ratings', 'user_predictions']
    
    temp = temp.loc[temp.user_ratings == 0]
    temp = temp.sort_values('user_predictions', ascending = False)
    #print('\n Below are the recommended places for user(user_id = {}):\n'. format(userID))
    #print(temp.head(num_recommendations))
    return temp.head(num_recommendations)
    

num_recommedations = 5

pivot_data.query('user_index')

u = []
b = []
fd = {}

for _ in range(len(final_ratings_matrix.index)):
    fd[final_ratings_matrix.index[_]] = None

for i in  range(len(pivot_data.query('user_index'))):
    x = recommend_places(i, pivot_data, pred_data, num_recommedations)

    for j in range(len(x)):
        b.append(x.index[j])
    print(b)
        

    fd[final_ratings_matrix.index[i]] = b
    u = []
    b = []


json_object = json.dumps(fd, indent = 4) 


with open('reccomended_restaurants.json', 'w') as json_file:
  json.dump(fd, json_file)
