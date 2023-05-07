import csv
from text_classification import classify
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

class Sentiment():

    """This method initializes an instance of the HotelRecommender class by creating an empty dictionary to store review data."""
    
    def __init__(self):
        self.rev_dat = {}
        
    """This method adds review data for a given business ID to the sentiment analysis dataset. 
        It takes two parameters: the business ID and the review, which is either 'positive', 'negative', or 'neutral'. 
        If the business ID is already present in self.rev_dat, the method updates the review count for the corresponding sentiment category. 
        If the business ID is not in self.rev_dat, the method creates a new entry with the corresponding review count."""

    def add_data(self, business_id, review):
        if business_id in self.rev_dat:
            if review == 'positive':
                self.rev_dat[business_id] = [self.rev_dat[business_id][0] + 1, self.rev_dat[business_id][1], self.rev_dat[business_id][2]]
            elif review == 'negative':
                self.rev_dat[business_id] = [self.rev_dat[business_id][0], self.rev_dat[business_id][1] + 1, self.rev_dat[business_id][2]]
            elif review == 'neutral':
                self.rev_dat[business_id] = [self.rev_dat[business_id][0], self.rev_dat[business_id][1], self.rev_dat[business_id][2] + 1]
            else:
                print("Wrong input for the business id: ", business_id, " inp (", review, ")")
        else:
            if review == 'positive':
                self.rev_dat[business_id] = [1, 0, 0]
            elif review == 'negative':
                self.rev_dat[business_id] = [0, 1, 0]
            elif review == 'neutral':
                self.rev_dat[business_id] = [0, 0, 1]
            else:
                print("Wrong input for the business id: ", business_id, " inp (", review, ")")

    """This method takes in a Pandas DataFrame row that contains a review text and the corresponding business ID. 
        It then uses a pre-trained machine learning model to classify the sentiment of the review text as either positive, negative or neutral. 
        The method returns a tuple containing the business ID and the predicted sentiment label."""
    
    def classify_review(self, row):
        return (row[1]['business_id'], str(classify(str(row[1]['text']))))
    
    """This method returns the current review data stored in the object. 
        The review data consists of a dictionary where each key is a business ID 
        and the value is a list of three integers representing the count of positive, negative, and neutral reviews for that business."""

    def commit_dat(self):
        return self.rev_dat

if __name__ == '__main__':
    s = Sentiment()
    print("Loading CSV File")
    df = pd.read_csv('top_reviews.csv')
    print("CSV File Loaded")
    for row in tqdm(df.iterrows(), total=len(df)):
        result = s.classify_review(row)
        s.add_data(result[0], result[1])

    s.commit_dat()
