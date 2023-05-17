from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from tqdm import tqdm
from gdrive_downloader import g_downloader


"""The combFeatures function takes in a single row of a Pandas DataFrame and returns a concatenated string of various features of the business represented in that row.
    At first, the function initializes the ret variable with the string value of the categories column of the input row. Then, the function checks the boolean values of several other columns in the row. 
    If the value is True, the corresponding feature is appended to the ret string with a space separator.
    Similarly, the function checks the integer values of several columns in the row, and if the value is 1, 
    the corresponding feature is also appended to the ret string with a space separator.
    After processing all the columns, the function returns the concatenated ret string.
    which provides a summary of the business's various attributes."""

def combFeatures(row):
    ret = str(row['categories']) 
    if row['RestaurantsTakeOut'] == "True":
        ret = ret + " " + "RestaurantsTakeOut"
    if row['BusinessAcceptsCreditCards'] == "True":
        ret = ret + " " + "BusinessAcceptsCreditCards"
    if row['RestaurantsDelivery'] == "True":
        ret = ret + " " + "RestaurantsDelivery"
    if row['BikeParking'] == "True":
        ret = ret + " " + "BikeParking"
    if row['Caters'] == "True":
        ret = ret + " " + "Caters"
    if row['GoodForKids'] == "True": 
        ret = ret + " " + "GoodForKids"
    if row['WheelchairAccessible'] == "True":
        ret = ret + " " + "WheelchairAccessible"
    if row['RestaurantsReservations'] == "True": 
        ret = ret + " " + "RestaurantsReservations"
    if row['HasTV'] == "True":
        ret = ret + " " + "HasTV"
    if row['RestaurantsGoodForGroups'] == "True":
       ret = ret + " " + "RestaurantsGoodForGroups"
    if row['RestaurantsTableService'] == "True": 
       ret = ret + " " + "RestaurantsTableService"
    if int(row['Restaurants']) == "1":
        ret = ret + " " + "Restaurants"
    if int(row['Food']) == "1":
        ret = ret + " " + "Food"
    if int(row["Pizza"]) == "1":
        ret = ret + " " + "Pizza"
    if int(row["Sandwiches"]) == "1":
        ret = ret + " " + "Sandwiches"
    if int(row['Nightlife']) == "1":
        ret = ret + " " + "Nightlife"
    if int (row['Bars']) == "1": 
        ret = ret + " " + "Bars"
    if int(row['Coffee & Tea']) == "1":
        ret = ret + " " + "Coffee & Tea"
    if int(row['American (Traditional)']) == "1":
        ret = ret + " " + "American (Traditional)"
    if int(row['Breakfast & Brunch']) == "1":
        ret = ret + " " + "Breakfast & Brunch"
    if int(row['Italian']) == "1": 
        ret = ret + " " + "Italian"
    if int(row['American (New)']) == "1":
        ret = ret + " " + "American (New)"
    if int(row['Specialty Food']) == "1":
        ret = ret + " " + "Specialty Food"
    if int(row['Burgers']) == "1": 
        ret = ret + " " + "Burgers"
    if int(row['Fast Food']) == "1": 
        ret = ret + " " + "Fast Food"
    if int(row['Event Planning & Services']) == "1":
        ret = ret + " " + "Event Planning & Services"
    if int(row['Shopping']) == "1":
        ret = ret + " " + "Shopping"
    if int(row['Chinese']) == "1":
        ret = ret + " " + "Chinese"
    if int(row['Grocery']) == "1":
        ret = ret + " " + "Grocery"
    if int(row['Bakeries']) == "1":
        ret = ret + " " + "Bakeries"
    if int(row['Seafood']) == "1":
        ret = ret + " " + "Seafood"
    return ret



class HotelRecommender:
    
    
    """This is the HotelRecommender class, and it takes in a few arguments. 
        The df_review is a pandas DataFrame that contains review data, and the sent is a sentiment analysis model that has already been trained.
        The csv_path argument is optional, but if provided, it's used to read in a CSV file that contains business data.
        The class has several attributes, including the df attribute, which is a pandas DataFrame that contains business data. The features attribute is a list of strings that represent different features of a business, such as whether it accepts credit cards or has bike parking. These features are used later on to help recommend hotels to users.
        The df_review and sent arguments are also attributes of the class, and are used to help with sentiment analysis of user reviews."""
    
    def __init__(self, df_review, sent, csv_path: str = "business_restaurant.csv"):
        print("Downloading Business Restaurent Data")
        g_downloader('1Ca0cBZ9BaoOeB2qAVMe2CO36aqT4G99x')
        print("Downloaded Business Restuarent Data")
        self.df = pd.read_csv(csv_path)
        self.features = ['categories', 'RestaurantsTakeOut', 'BusinessAcceptsCreditCards', 'RestaurantsDelivery',
                         'BikeParking', 'Caters', 'GoodForKids', 'WheelchairAccessible', 'RestaurantsReservations',
                         'HasTV', 'RestaurantsGoodForGroups', 'RestaurantsTableService', 'Restaurants', 'Food', 'Pizza',
                         'Sandwiches', 'Nightlife', 'Bars', 'Coffee & Tea', 'American (Traditional)',
                         'Breakfast & Brunch', 'Italian', 'American (New)', 'Specialty Food', 'Burgers', 'Fast Food',
                         'Event Planning & Services', 'Shopping', 'Chinese', 'Grocery', 'Bakeries', 'Seafood']
        self.df_review = df_review
        self.sent = sent
        
    """ This function, called getTitle, takes in an index value and returns a tuple that contains the name, 
        a dictionary of features, business_id, stars and review count of the corresponding business in the dataframe.

        The index parameter should be a valid integer index value of the dataframe.

        The function returns a tuple that contains the following elements in order:

        name: a string that represents the name of the business
        features: a dictionary that contains all the features of the business represented in the input row
        business_id: a string that represents the unique identifier of the business
        stars: a float value that represents the average rating of the business
        review_count: an integer value that represents the number of reviews of the business"""      
        
    def getTitle(self, index):
        return self.df[self.df.index == index]["name"].values[0], self.df[self.df.index == index].to_dict('list'),self.df[self.df.index == index]["business_id"].values[0], self.df[self.df.index == index]["stars"].values[0],self.df[self.df.index == index]["review_count"].values[0]

    """This function takes in a hotel title and returns its corresponding index in the pandas DataFrame. 
    If the title is not found, it returns -1."""
    
    def getIndex(self, title):
        if str((self.df[self.df.name == title])).startswith("Empty"):
            return -1
        return self.df[self.df.name == title]["index"].values[0]
    
    """This method recommends hotels based on a given hotel and number of recommendations. 
        It uses cosine similarity to determine similarity between hotel features and recommends hotels based on that. 
        It also filters reviews and calculates a weighted score for each hotel based on similarity, stars, review count, and sentiment data. 
        The hotels are then sorted by their weighted scores in descending order and the top recommended hotels are returned"""
        
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
            print("Sorry the Business was not found")
            return -1
        hotelRetCount = recommendation_num
        similar_hotels = list(enumerate(similarityElement[hotelIndex]))
        sortedSimilar = sorted(similar_hotels, key=lambda x: x[1], reverse=True)[1:]
        i = 0
        list_df = []
        if len(sortedSimilar) < 1:
            return 0
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
                if i == hotelRetCount:
                    break
                
            df_2 = pd.DataFrame(list_df)
            
            #Filter Reviews
            df_filtered = self.df_review[self.df_review['business_id'].isin(listed_business_ids)]
            # Group the reviews by business ID, and get the top 10 most useful reviews for each group
            top_reviews = (df_filtered.groupby('business_id')
                        .apply(lambda x: x.nlargest(5, 'useful'))
                        .reset_index(drop=True))

            for row in tqdm(top_reviews.iterrows(), total=len(top_reviews)):
                result = self.sent.classify_review(row)
                self.sent.add_data(result[0], result[1])
            rev_data = self.sent.commit_dat()
            
            # Calculate the weighted score for each business ID and store it in a dictionary
            business_scores = {}
            for i in range(len(listed_business_ids)):
                business_id = listed_business_ids[i]
                business_name = out[i]
                stars = list_stars[i]
                review_count = list_review_count[i]
                similarity = list_similarity[i]
                senti_data = rev_data[business_id]
                weighted_score = 0.5 * similarity +  0.3 * (stars * review_count) + 0.2 * (senti_data[0]/5)
                business_scores[business_name] = weighted_score

            print(business_scores)
            # Sort the business IDs by their weighted scores in descending order
            sorted_business_ids = sorted(business_scores.keys(), key=lambda x: business_scores[x], reverse=True)
            original_indices = [list(business_scores.keys()).index(x) for i, x in enumerate(sorted_business_ids) if x in list(business_scores.keys())]
            print(sorted_business_ids)
            print(original_indices)
            # Print the sorted business IDs
            #print(sorted_business_ids[:hotelRetCount])
            print("Recommended Businesses are \n ")
            
            for i in original_indices:
                print("Business Name: ", out[i])
                print("Stars: ", list_stars[i])
                print("Review Count: ", list_review_count[i])
                print("Combined Features: ", similar_features[i])
                print("Weightage Scores: ", business_scores[out[i]])
            
            
            return sorted_business_ids[:hotelRetCount]
