#Import the train_test_split function

from sklearn.model_selection import train_test_split
#The train_test_split function splits data into training and testing sets while preserving the relationships between feature variables and the target variable.

#Define the Feature and Target Columns

feature_column_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_preb', 'age']
predicted_class_name = ['diabetes']

##feature_column_names: A list of columns in the dataset that will be used as input features (X).
##predicted_class_name: The column name in the dataset that represents the target variable (Y).

##Extract Feature (𝑋 and Target (y) Data

X = data_frame[feature_column_names].values
y = data_frame[predicted_class_name].values

##X: Feature matrix containing the independent variables. .values converts the selected columns into a NumPy array.
##y: Target array containing the dependent variable(s), also converted into a NumPy array.

split_test_size = 0.30

