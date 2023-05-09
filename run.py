from recommender import HotelRecommender
import pandas as pd
from review_classification import Sentiment

print("loading Review Dataframe")
df_review = pd.read_csv('/Users/shilpashivarudraiah/Yelp_Dataset/business_review_pa.csv')
print("Review Dataframe Loaded")
sent = Sentiment()
s = HotelRecommender(df_review, sent, 'business_restaurant.csv')
rel = []
numb = 10
while True:
    user_input = input("Enter a business name to get recommendations or press 'q' to exit: ")
    if user_input == 'q':
        break
    
    _ = s.recommend(user_input, numb)
    if _ is not -1:
        valid = input("Please enter the number of recommendations which you find relevant: ")
        rel.append(int(valid))
        print("Precision is: ", round((int(valid)/numb),2))

map = 0
for i in rel:
    map+=i/numb
map=map/len(rel)
print("Mean Average Precision is: ", round(map,2))

