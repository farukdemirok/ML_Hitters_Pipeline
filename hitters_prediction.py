################################################
# End-to-End Hitters Machine Learning Pipeline III
################################################

# 6. Prediction for new observation

import joblib
import pandas as pd


df = pd.read_csv("HAFTA_08/Ders Notları/hitters.csv")


random_user = df.sample(1, random_state=45)
new_model = joblib.load("HAFTA_08/voting_clf_hitters.pkl")

new_model.predict(random_user)


from HAFTA_08.hitters_pipeline import *

X, y = hitters_data_prep(df)

random_user = X.sample(1, random_state=45)
new_model = joblib.load("HAFTA_08/voting_clf_hitters.pkl")
new_model.predict(random_user)



