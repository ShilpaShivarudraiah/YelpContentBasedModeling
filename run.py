from recommender import HotelRecommender
import pandas as pd
print("loading Review Dataframe")
# df_review = pd.read_csv('yelp_academic_dataset_review.csv')
print("Review Dataframe Loaded")
s = HotelRecommender("df_review", 'business_restaurant.csv')
while(True):
    s.recommend( 'PrimoHoagies', 5 )
    break