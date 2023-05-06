from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def combFeatures(row):
    ret = str(row['categories']) 
    if row['RestaurantsTakeOut'] == True:
        ret = ret + " " + "RestaurantsTakeOut"
    if row['BusinessAcceptsCreditCards'] == True:
        ret = ret + " " + "BusinessAcceptsCreditCards"
    if row['RestaurantsDelivery'] == True:
        ret = ret + " " + "RestaurantsDelivery"
    if row['BikeParking'] == True:
        ret = ret + " " + "BikeParking"
    if row['Caters'] == True:
        ret = ret + " " + "Caters"
    if row['GoodForKids'] == True: 
        ret = ret + " " + "GoodForKids"
    if row['WheelchairAccessible'] == True:
        ret = ret + " " + "WheelchairAccessible"
    if row['RestaurantsReservations'] == True: 
        ret = ret + " " + "RestaurantsReservations"
    if row['HasTV'] == True:
        ret = ret + " " + "HasTV"
    if row['RestaurantsGoodForGroups'] == True:
       ret = ret + " " + "RestaurantsGoodForGroups"
    if row['RestaurantsTableService'] == True: 
       ret = ret + " " + "RestaurantsTableService"
    if int(row['Restaurants']) == 1:
        ret = ret + " " + "Restaurants"
    if int(row['Food']) == 1:
        ret = ret + " " + "Food"
    if int(row["Pizza"]) == 1:
        ret = ret + " " + "Pizza"
    if int(row["Sandwiches"]) == 1:
        ret = ret + " " + "Sandwiches"
    if int(row['Nightlife']) == 1:
        ret = ret + " " + "Nightlife"
    if int (row['Bars']) == 1: 
        ret = ret + " " + "Bars"
    if int(row['Coffee & Tea']) == 1:
        ret = ret + " " + "Coffee & Tea"
    if int(row['American (Traditional)']) == 1:
        ret = ret + " " + "American (Traditional)"
    if int(row['Breakfast & Brunch']) == 1:
        ret = ret + " " + "Breakfast & Brunch"
    if int(row['Italian']) == 1: 
        ret = ret + " " + "Italian"
    if int(row['American (New)']) == 1:
        ret = ret + " " + "American (New)"
    if int(row['Specialty Food']) == 1:
        ret = ret + " " + "Specialty Food"
    if int(row['Burgers']) == 1: 
        ret = ret + " " + "Burgers"
    if int(row['Fast Food']) == 1: 
        ret = ret + " " + "Fast Food"
    if int(row['Event Planning & Services']) == 1:
        ret = ret + " " + "Event Planning & Services"
    if int(row['Shopping']) == 1:
        ret = ret + " " + "Shopping"
    if int(row['Chinese']) == 1:
        ret = ret + " " + "Chinese"
    if int(row['Grocery']) == 1:
        ret = ret + " " + "Grocery"
    if int(row['Bakeries']) == 1:
        ret = ret + " " + "Bakeries"
    if int(row['Seafood']) == 1:
        ret = ret + " " + "Seafood"
    return ret



class HotelRecommender:
    def __init__(self, df_review, csv_path: str = "business_restaurant.csv"):
        self.df = pd.read_csv(csv_path)
        self.features = ['categories', 'RestaurantsTakeOut', 'BusinessAcceptsCreditCards', 'RestaurantsDelivery',
                         'BikeParking', 'Caters', 'GoodForKids', 'WheelchairAccessible', 'RestaurantsReservations',
                         'HasTV', 'RestaurantsGoodForGroups', 'RestaurantsTableService', 'Restaurants', 'Food', 'Pizza',
                         'Sandwiches', 'Nightlife', 'Bars', 'Coffee & Tea', 'American (Traditional)',
                         'Breakfast & Brunch', 'Italian', 'American (New)', 'Specialty Food', 'Burgers', 'Fast Food',
                         'Event Planning & Services', 'Shopping', 'Chinese', 'Grocery', 'Bakeries', 'Seafood']
        self.df_review = df_review
    def getTitle(self, index):
        return self.df[self.df.index == index]["name"].values[0], self.df[self.df.index == index].to_dict('list'),self.df[self.df.index == index]["business_id"].values[0], self.df[self.df.index == index]["stars"].values[0],self.df[self.df.index == index]["review_count"].values[0]

    def getIndex(self, title):
        if str((self.df[self.df.name == title])).startswith("Empty"):
            return -1
        return self.df[self.df.name == title]["index"].values[0]
    

    def recommend(self, users_hotel: str = "", recommendation_num: int = 10):
        self.df['index'] = self.df.reset_index().index
        for feature in self.features:
            if self.df[feature].dtype == bool:
                self.df[feature] = self.df[feature].fillna(False)
            else:
                self.df[feature] = self.df[feature].fillna(0)
        self.df["combinedFeatures"] = self.df.apply(combFeatures, axis=1)
        
        cv = CountVectorizer()
        countMatrix = cv.fit_transform(self.df["combinedFeatures"])
        similarityElement = cosine_similarity(countMatrix)
        hotelIndex = self.getIndex(users_hotel)

        if hotelIndex == -1:
            return -1
        hotelRetCount = recommendation_num
        similar_hotels = list(enumerate(similarityElement[hotelIndex]))
        sortedSimilar = sorted(similar_hotels, key=lambda x: x[1], reverse=True)[1:]
        i = 0
        list_df = []
        if len(sortedSimilar) < 1:
            return -1
        else:
            out = []
            similar_features = []
            listed_business_ids = []
            list_stars = []
            list_review_count = []
            list_similarity = []
            for element in sortedSimilar:
                i = i + 1
                list_similarity.append(element[1])
                recommended_restaurant,temp_df,business_id,stars,review_count = self.getTitle(element[0])
                # Append the original DataFrame to the new DataFrame
                list_df.append(temp_df)
                list_stars.append(stars)
                list_review_count.append(review_count)
                listed_business_ids.append(business_id)
                out.append(recommended_restaurant)
                recommended_features = self.df.loc[self.df['name'] == recommended_restaurant, 'combinedFeatures'].iloc[0]
                similar_features.append(recommended_features)
                if i == 100:
                    break
                
            df_2 = pd.DataFrame(list_df)
            
            # df_2.info()
            # print(list_stars)
            # print(list_review_count)
            # print(list_similarity)
            
            # Calculate the weighted score for each business ID and store it in a dictionary
            business_scores = {}
            for i in range(len(listed_business_ids)):
                business_id = out[i]
                stars = list_stars[i]
                review_count = list_review_count[i]
                similarity = list_similarity[i]
                weighted_score = 0.5 * (similarity + stars * review_count)
                business_scores[business_id] = weighted_score

            # Sort the business IDs by their weighted scores in descending order
            sorted_business_ids = sorted(business_scores.keys(), key=lambda x: business_scores[x], reverse=True)

            # Print the sorted business IDs
            print(sorted_business_ids[:hotelRetCount])
            
            
            # #Filter Reviews
            # df_filtered = self.df_review[self.df_review['business_id'].isin(listed_business_ids)]
            # # Group the reviews by business ID, and get the top 10 most useful reviews for each group
            # top_reviews = (df_filtered.groupby('business_id')
            #             .apply(lambda x: x.nlargest(5, 'useful'))
            #             .reset_index(drop=True))

            # # Print the resulting DataFrame
            # top_reviews.to_csv('top_reviews.csv')
            return sorted_business_ids[:hotelRetCount]
