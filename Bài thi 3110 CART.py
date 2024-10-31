import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Đường dẫn tới thư mục chứa ảnh, ở đây chúng ta thực hiện lưu trong ổ D
image_dir = r'D:\Bài 31.10.2024\tooth'

# Lấy danh sách các tệp tin trong thư mục
file_list = [f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]

# Đọc ảnh và gán nhãn
def load_images_and_labels(file_list, folder_path):
    images = []
    labels = []
    for file_name in file_list:
        img_path = os.path.join(folder_path, file_name)
        with Image.open(img_path) as img:
            img_array = np.array(img.resize((100, 100)))  # Resize ảnh
            images.append(img_array.flatten())  # Chuyển đổi thành vector 1D
            labels.append(1 if "lesion" in file_name.lower() else 0)  # Gán nhãn
    return np.array(images), np.array(labels)

# Đọc ảnh và gán nhãn
X_dental, y_dental = load_images_and_labels(file_list, image_dir)
print(f'Đã đọc {len(X_dental)} ảnh từ thư mục {image_dir}')

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train_dental, X_test_dental, y_train_dental, y_test_dental = train_test_split(X_dental, y_dental, test_size=0.2, random_state=42)

# CART (Gini Index)
cart_model_dental = DecisionTreeClassifier(criterion='gini')
cart_model_dental.fit(X_train_dental, y_train_dental)

# Dự đoán và đánh giá mô hình CART
cart_predictions_dental = cart_model_dental.predict(X_test_dental)
print("CART Model (Dental) Accuracy:", accuracy_score(y_test_dental, cart_predictions_dental))
print("CART Classification Report (Dental):\n", classification_report(y_test_dental, cart_predictions_dental))

# ID3 (Information Gain)
id3_model_dental = DecisionTreeClassifier(criterion='entropy')
id3_model_dental.fit(X_train_dental, y_train_dental)

# Dự đoán và đánh giá mô hình ID3
id3_predictions_dental = id3_model_dental.predict(X_test_dental)
print("ID3 Model (Dental) Accuracy:", accuracy_score(y_test_dental, id3_predictions_dental))
print("ID3 Classification Report (Dental):\n", classification_report(y_test_dental, id3_predictions_dental))
