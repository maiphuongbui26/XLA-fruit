import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk  

# === Bước 1: Tải dữ liệu hình ảnh và gán nhãn ===
def load_images_from_folder(folder):
    images = []
    labels = []
    sizes = []  # Danh sách lưu kích thước
    for filename in os.listdir(folder):
        if "_" in filename and filename.endswith((".jpg", ".png", ".jpeg")):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                # Lấy nhãn từ tên tệp
                try:
                    label = filename.split("_")[1].split(".")[0]
                    labels.append(label)
                    # Thêm kích thước của quả
                    size = get_fruit_size(img)
                    sizes.append(size)
                except IndexError:
                    print(f"Tệp không có nhãn hợp lệ: {filename}")
                    continue
        else:
            print(f"Bỏ qua tệp không hợp lệ: {filename}")
    return images, labels, sizes

def get_fruit_size(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([30, 40, 40])
    upper_color = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        return area
    else:
        return 0

def classify_size(area):
    if area < 1000:
        return "nhỏ"
    elif area < 3000:
        return "vừa"
    else:
        return "lớn"

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

def create_dataframe(features, labels, sizes):
    df = pd.DataFrame(features)
    df['label'] = labels
    df['size'] = sizes
    df['size_category'] = df['size'].apply(classify_size)
    return df

def analyze_data(df):
    if df['label'].nunique() > 1:
        sns.pairplot(df, hue="label")
        plt.show()
    else:
        print("Không đủ nhãn khác nhau để hiển thị biểu đồ.")

def train_model(df):
    X = df[['mean_hue', 'mean_saturation', 'mean_value', 'size']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model

def predict_image_status(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể tải ảnh từ {image_path}")
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    size = get_fruit_size(img)
    features = pd.DataFrame([[np.mean(h), np.mean(s), np.mean(v), size]],
                            columns=['mean_hue', 'mean_saturation', 'mean_value', 'size'])
    predicted_label = model.predict(features)[0]
    predicted_size_category = classify_size(size)
    return predicted_label, predicted_size_category, img


# Giao diện người dùng
class FruitClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("Fruit Classifier")

        # Đường dẫn thư mục huấn luyện mặc định
        self.folder_path = "C:/Users/ACER/Downloads/asm/img"  # Thay bằng đường dẫn thực tế

        self.label = tk.Label(master, text="Chọn thư mục kiểm tra:")
        self.label.pack()

        self.test_button = tk.Button(master, text="Chọn thư mục kiểm tra", command=self.load_test_folder)
        self.test_button.pack()

        self.predict_button = tk.Button(master, text="Dự đoán", command=self.predict)
        self.predict_button.pack()

        self.test_image_paths = []
        self.model = None

        # Tự động tải thư mục huấn luyện khi khởi động
        self.load_folder()

    def load_folder(self):
        if os.path.exists(self.folder_path):
            images, labels, sizes = load_images_from_folder(self.folder_path)
            hsv_images = convert_to_hsv(images)
            features = extract_color_features(hsv_images)
            df = create_dataframe(features, labels, sizes)
            self.model = train_model(df)
            messagebox.showinfo("Thông báo", "Đã tải và huấn luyện mô hình thành công!")
        else:
            messagebox.showerror("Lỗi", f"Thư mục huấn luyện không tồn tại: {self.folder_path}")

    def load_test_folder(self):
        self.test_image_paths = filedialog.askdirectory()
        if self.test_image_paths:
            # Lấy tất cả các tệp ảnh trong thư mục đã chọn
            self.test_image_paths = [os.path.join(self.test_image_paths, f) for f in os.listdir(self.test_image_paths) 
                                     if f.endswith((".jpg", ".jpeg", ".png"))]
            messagebox.showinfo("Thông báo", f"Đã chọn {len(self.test_image_paths)} ảnh để kiểm tra!")

    def predict(self):
        if not self.test_image_paths or self.model is None:
            messagebox.showerror("Lỗi", "Vui lòng chọn thư mục chứa ảnh kiểm tra và đảm bảo mô hình đã được huấn luyện.")
            return
        results = []
        for image_path in self.test_image_paths:
            status, size_category, img = predict_image_status(image_path, self.model)
            if status:
                if status == "rotten":
                    status = "hỏng"
                elif status == "ripe":
                    status = "chín"
                results.append([image_path, status, size_category, img])
            else:
                results.append([image_path, "Không thể nhận diện", "", None])

        self.show_results(results)

    def show_results(self, results):
        # Tạo cửa sổ mới để hiển thị kết quả
        result_window = tk.Toplevel(self.master)
        result_window.title("Bảng kết quả dự đoán")

        # Tạo Treeview để hiển thị bảng
        tree = ttk.Treeview(result_window, columns=("Image", "Status", "Size"), show="headings")
        tree.pack(fill=tk.BOTH, expand=True)

        # Định nghĩa các cột
        tree.heading("Image", text="Ảnh")
        tree.heading("Status", text="Trạng thái")
        tree.heading("Size", text="Kích thước")

        # Thêm kết quả vào bảng
        for result in results:
            img = result[3]
            if img is not None:
                # Chuyển ảnh thành thumbnail
                thumbnail = self.create_thumbnail(img)
                photo = ImageTk.PhotoImage(thumbnail)
                tree.insert("", tk.END, values=(None, result[1], result[2]), image=photo)
            else:
                tree.insert("", tk.END, values=(result[0], result[1], result[2]))

        # Cung cấp thanh cuộn dọc cho Treeview
        scrollbar = tk.Scrollbar(result_window, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)

    def create_thumbnail(self, img, size=(100, 100)):
        """Tạo ảnh thu nhỏ cho việc hiển thị trong Treeview."""
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img.thumbnail(size)
        return img

# Chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = FruitClassifierApp(root)
    root.mainloop()
