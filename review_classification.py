import csv
from text_classification import classify
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

class Sentiment():

    def __init__(self):
        self.rev_dat = {}

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

    def classify_review(self, row):
        return (row[1]['business_id'], str(classify(str(row[1]['text']))))

    def commit_dat(self, path_of_datafile: str = "/content/drive/MyDrive/buss_po_neg_neu.csv"):
        with open(path_of_datafile, "w") as datafile:
            for key in self.rev_dat:
                if key == "business_id":
                    datafile.write("business_id, positive_count, negative_count, neutral_count")
                else:
                    datafile.write(f"\n{key}, {self.rev_dat[key][0]}, {self.rev_dat[key][1]}, {self.rev_dat[key][2]}")

if __name__ == '__main__':
    s = Sentiment()
    print("Loading CSV File")
    df = pd.read_csv('top_reviews.csv')
    print("CSV File Loaded")
    for row in tqdm(df.iterrows(), total=len(df)):
        result = s.classify_review(row)
        s.add_data(result[0], result[1])

    s.commit_dat()
