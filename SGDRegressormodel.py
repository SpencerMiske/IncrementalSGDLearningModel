# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Corrected the typo here
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

class IncrementalRegressor:

    def __init__(self, filePath, indVar, initialTests):
        # Open the Excel file
        self.filePath = filePath
        self.file = pd.read_excel(self.filePath, sheet_name='Sheet1')

        # Initialize arrays for X and Y values
        self.variables = indVar
        self.initial_x = []
        self.initial_y = []

        # Populate initial_x and initial_y with data from the file
        for i in range(initialTests):
            self.initial_y.append(self.file.iloc[i, 0])
            temp = []
            for y in range(indVar):
                temp.append(self.file.iloc[i, y + 1])
            self.initial_x.append(temp)

        # Convert lists to NumPy arrays
        self.initial_x = np.array(self.initial_x)
        self.initial_y = np.array(self.initial_y)

        # Scale the data
        self.scale = StandardScaler()
        self.scaled_x = self.scale.fit_transform(self.initial_x)

        # Initialize SGD regressor model with initial data
        self.sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)
        self.sgd_model.fit(self.scaled_x, self.initial_y)

    def update_model(self, x, y):
        # Scale the new data
        new_x_scaled = self.scale.transform(np.array(x).reshape(1, -1))

        # Train the model with the new data
        self.sgd_model.partial_fit(new_x_scaled, np.array([y]))

        # Append the new data to the existing data
        self.scaled_x = np.vstack([self.scaled_x, new_x_scaled])
        self.initial_y = np.append(self.initial_y, y)

        # Predict and evaluate using all data
        y_init_predict = self.sgd_model.predict(self.scaled_x)
        self.mse = mean_squared_error(self.initial_y, y_init_predict)

        print(f'Updated MSE: {self.mse:.4f}')

    def predict(self, x):
        # Scale the input data
        x_scaled = self.scale.transform(np.array(x).reshape(1, -1))

        # Predict the output
        y_predict = self.sgd_model.predict(x_scaled)

        print('Predicted output:', y_predict[0])
