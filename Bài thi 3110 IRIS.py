import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Tải bộ dữ liệu IRIS
iris = load_iris()
X_iris = iris.data  
# Các đặc trưng
y_iris = iris.target  
# Nhãn

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# CART (Gini Index)
cart_model = DecisionTreeClassifier(criterion='gini')
cart_model.fit(X_train_iris, y_train_iris)

# Dự đoán và đánh giá mô hình CART
cart_predictions_iris = cart_model.predict(X_test_iris)
print("CART Model (IRIS) Accuracy:", accuracy_score(y_test_iris, cart_predictions_iris))
print("CART Classification Report (IRIS):\n", classification_report(y_test_iris, cart_predictions_iris))

# ID3 (Information Gain)
id3_model = DecisionTreeClassifier(criterion='entropy')
id3_model.fit(X_train_iris, y_train_iris)

# Dự đoán và đánh giá mô hình ID3
id3_predictions_iris = id3_model.predict(X_test_iris)
print("ID3 Model (IRIS) Accuracy:", accuracy_score(y_test_iris, id3_predictions_iris))
print("ID3 Classification Report (IRIS):\n", classification_report(y_test_iris, id3_predictions_iris))
