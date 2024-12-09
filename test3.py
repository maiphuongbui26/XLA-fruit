import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# === Bước 1: Tải dữ liệu hình ảnh và gán nhãn ===
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if "_" in filename and filename.endswith((".jpg", ".png", ".jpeg")):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                # Lấy nhãn từ tên tệp"
                try:
                    labels.append(filename.split("_")[1].split(".")[0])
                except IndexError:
                    print(f"Tệp không có nhãn hợp lệ: {filename}")
                    continue
        else:
            print(f"Bỏ qua tệp không hợp lệ: {filename}")
    return images, labels

# === Bước 2: Chuyển đổi không gian màu và trích xuất đặc trưng ===
def convert_to_hsv(images):
    return [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in images]

def extract_color_features(hsv_images):
    features = []
    for img in hsv_images:
        h, s, v = cv2.split(img)
        features.append({
            "mean_hue": np.mean(h),
            "mean_saturation": np.mean(s),
            "mean_value": np.mean(v)
        })
    return features

# === Bước 3: Tạo DataFrame và phân tích dữ liệu ===
def create_dataframe(features, labels):
    df = pd.DataFrame(features)
    df['label'] = labels
    return df


def analyze_data(df):
    # Kiểm tra có đủ nhãn để vẽ biểu đồ hay không
    if df['label'].nunique() > 1:
        sns.pairplot(df, hue="label")
        plt.show()
    else:
        print("Không đủ nhãn khác nhau để hiển thị biểu đồ.")


# === Bước 4: Huấn luyện mô hình ===
def train_model(df):
    X = df[['mean_hue', 'mean_saturation', 'mean_value']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model

# === Bước 5: Nhận diện trạng thái hoa quả từ ảnh mới ===
def predict_image_status(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể tải ảnh từ {image_path}")
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    features = [[np.mean(h), np.mean(s), np.mean(v)]]
    return model.predict(features)[0]

# Chạy
if __name__ == "__main__":
    # Thư mục chứa ảnh huấn luyện (thay)
    folder = "asm/img"
    
    # 1. Tải dữ liệu
    images, labels = load_images_from_folder(folder)
    print("Số lượng ảnh:", len(images))
    
    # 2. Chuyển đổi HSV và trích xuất đặc trưng
    hsv_images = convert_to_hsv(images)
    features = extract_color_features(hsv_images)
    
    # 3. Tạo DataFrame và phân tích
    df = create_dataframe(features, labels)
    print(df.head())
    analyze_data(df)
    
    # 4. Huấn luyện mô hình
    model = train_model(df)
    
    # 5. Nhận diện trạng thái từ ảnh mới
    test_image_path = "asm/img_test/test3.jpg"  # Đường dẫn ảnh cần nhận diện (thay)
    status = predict_image_status(test_image_path, model)
    if status:
        print("Trạng thái hoa quả:", status)
    else:
        print("Không thể nhận diện trạng thái.")