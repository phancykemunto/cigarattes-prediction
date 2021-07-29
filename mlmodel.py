import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("C:/PYDATAFILES/CIGARATTES.csv")



#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model.pkl','wb'))