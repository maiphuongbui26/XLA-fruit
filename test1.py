import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

# === Xử lý dữ liệu ===
def taithumuc(folder):
    """
    Tải hình ảnh và gán nhãn từ thư mục.
    """
    images, labels, sizes = [], [], []
    for filename in os.listdir(folder):
        if "_" in filename and filename.endswith((".jpg", ".png", ".jpeg")):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                label = trichxuatnhan(filename)
                if label:
                    images.append(img)
                    labels.append(label)
                    sizes.append(tinh_kichthuoc(img))
    return images, labels, sizes

def trichxuatnhan(filename):
    """
    Trích xuất nhãn từ tên tệp.
    """
    try:
        return filename.split("_")[1].split(".")[0]
    except IndexError:
        print(f"Tệp không có nhãn hợp lệ: {filename}")
        return None

def tinh_kichthuoc(image):
    """
    Tính kích thước của quả dựa trên diện tích vùng phát hiện.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cv2.contourArea(contour) for contour in contours) if contours else 0

def phanloai_kichthuoc(area):
    """
    Phân loại kích thước theo diện tích.
    """
    if area < 1000:
        return "nhỏ"
    elif area < 3000:
        return "vừa"
    return "lớn"

# === Xử lý đặc trưng ===
def chuyendoimau_hsv(images):
    return [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in images]

def trichxuatmau_hsv(hsv_images):
    """
    Trích xuất các đặc trưng màu sắc từ hình ảnh HSV.
    """
    return [{"mean_hue": np.mean(h), "mean_saturation": np.mean(s), "mean_value": np.mean(v)}
            for h, s, v in (cv2.split(img) for img in hsv_images)]

def tao_bangdulieu(features, labels, sizes):
    """
    Tạo DataFrame từ đặc trưng, nhãn và kích thước.
    """
    df = pd.DataFrame(features)
    df["label"] = labels
    df["size"] = sizes
    df["size_category"] = df["size"].apply(phanloai_kichthuoc)
    return df

# === Phân tích và huấn luyện mô hình ===
def phantich_dulieu(df):
    """
    Phân tích dữ liệu qua biểu đồ.
    """
    if df["label"].nunique() > 1:
        sns.pairplot(df, hue="label")
        plt.show()
    else:
        print("Không đủ nhãn khác nhau để hiển thị biểu đồ.")

def huanluyen_mohinh(df):
    """
    Huấn luyện mô hình Random Forest.
    """
    X = df[["mean_hue", "mean_saturation", "mean_value", "size"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))
    return model


def dudoanketqua(image_path, model):
    """
    Dự đoán nhãn và kích thước của một hình ảnh.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể tải ảnh từ {image_path}")
        return None, None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    size = tinh_kichthuoc(img)

    # Tạo DataFrame với tên cột giống như khi huấn luyện
    features = pd.DataFrame([{
        "mean_hue": np.mean(h),
        "mean_saturation": np.mean(s),
        "mean_value": np.mean(v),
        "size": size
    }])

    predicted_label = model.predict(features)[0]
    predicted_size_category = phanloai_kichthuoc(size)
    return predicted_label, predicted_size_category


# === Giao diện người dùng ===
class FruitClassifierApp:
    def __init__(self, master):
        self.master = master
        self.model = None
        self.test_image_paths = []
        self.setup_ui()

    def setup_ui(self):
        """
        Cài đặt giao diện người dùng.
        """
        self.master.title("Fruit Classifier")
        tk.Label(self.master, text="Chọn thư mục huấn luyện:").pack()
        tk.Button(self.master, text="Tải thư mục", command=self.tai_thumuc_huanluyen).pack()
        tk.Label(self.master, text="Chọn thư mục kiểm tra:").pack()
        tk.Button(self.master, text="Tải thư mục kiểm tra", command=self.tai_thumuc_kiemtra).pack()
        tk.Button(self.master, text="Dự đoán", command=self.predict).pack()
        self.result_label = tk.Label(self.master, text="")
        self.result_label.pack()

    def tai_thumuc_huanluyen(self):
        folder = filedialog.askdirectory()
        if folder:
            images, labels, sizes = taithumuc(folder)
            hsv_images = chuyendoimau_hsv(images)
            features = trichxuatmau_hsv(hsv_images)
            df = tao_bangdulieu(features, labels, sizes)
            self.model = huanluyen_mohinh(df)
            messagebox.showinfo("Thông báo", "Mô hình đã được huấn luyện!")

    def tai_thumuc_kiemtra(self):
        folder = filedialog.askdirectory()
        if folder:
            self.test_image_paths = [os.path.join(folder, f) for f in os.listdir(folder) 
                                     if f.endswith((".jpg", ".jpeg", ".png"))]
            messagebox.showinfo("Thông báo", f"Đã chọn {len(self.test_image_paths)} ảnh kiểm tra!")

    def predict(self):
        if not self.test_image_paths or self.model is None:
            messagebox.showerror("Lỗi", "Vui lòng chọn dữ liệu kiểm tra và đảm bảo mô hình đã huấn luyện.")
            return
        results = [dudoanketqua(img, self.model) for img in self.test_image_paths]
        self.result_label.config(text="\n".join(f"{status} ({size})" for status, size in results if status))

# === Chạy ứng dụng ===
if __name__ == "__main__":
    root = tk.Tk()
    FruitClassifierApp(root)
    root.mainloop()