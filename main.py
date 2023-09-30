import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from model import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Lưu ý input data phải ở dạng numpy array
# Các cột rời rạc phải được số hóa

# define model với min_samples = 2, max_dept = 2
model = DecisionTree(2,2)

# Tiền xử lý
penguins = load_penguins()
penguins=pd.get_dummies(penguins, columns = ["island"], prefix = ["island"]) # one hot encoding
species_mapping = {'Adelie': 0, 'Gentoo': 1, 'Chinstrap': 2}  
sex_mapping = {'male': 0, 'female': 1} 

penguins["species"] = penguins["species"].replace(species_mapping) 
penguins["sex"] = penguins["sex"].replace(sex_mapping)

penguins = np.array(penguins)
X = penguins[:,1:]
y = penguins[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22520691)
y_train = np.expand_dims(y_train, axis=1)
y_test = np.expand_dims(y_test, axis=1)

# Train và evalutions
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
