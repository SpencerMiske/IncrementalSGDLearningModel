#Before runnning code install "scikit Learn" and "numpy" and "pandas openyxl" in command center

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandaardScalar
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error


class incrementalRegressor:

    def __init__(self, filePath, tol, indVar, initialTests):

        #open excel file
        self.filePath = filePath
        self.file = pd.read_excel(self.filePath, sheet_name='sheet1')

        #get x and y values into array
        self.variables = indVar
        self.initial_x = np.empty([initialTests, indVar])
        self.initial_y = np.empty([initialTests])
        
        for i in range(initialTests):
            self.initial_y.append(self.file.iloc[i,0])
            temp = np.empty([indVar])
            for y in range(indVar):
                temp.append(self.file.iloc[i,y+1])
            self.initial_x.append(temp)

        #Scale the data
        self.scale = StandardScaler()
        self.scaled_x = self.scale.fit_transform(self.initial_x)

        #Initia;ize sgd regressor model with initial data
        self.sgd_model = SGDRegressor(max_it = 1000, tol=1e-3)
        self.sgd_model.fit(self.scaled_x, self.initial_y.ravel())

    def update_model(self, x, y):

        #Scale data
        new_x_scaled = self.scale.transorm(np.array(x).reshape(1,-1))

        #Train model with data
        self.sgd_model.partial_fit(new_x_scaled, np.array(x), np.array(y).reshape(1,))


        #Adds inputted value to initial values
        self.scaled_x.append(new_x_scaled)
        self.initial_y.append(y)
    
        #predict and evaluate using all data
        y_init_predict = self.sgd_model.predict(scaled_x)
        self.mse = mean_squared_error(self.initial_y, y_init_predict)

        print('Updated MSE: {mse:.4f}')

    def predict(self, x):

        x_scaled = self.scale.transform(np.array(x).reshape(1,-1))
        
        y_predict = self.sgd_model.predict(x_scaled)

        print('predicted output: ' + y_predict)
