import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageTk

# === Các hàm xử lý dữ liệu ===
def taithumuc(folder):
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
    try:
        return filename.split("_")[1].split(".")[0]
    except IndexError:
        print(f"Tệp không có nhãn hợp lệ: {filename}")
        return None

def tinh_kichthuoc(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cv2.contourArea(contour) for contour in contours) if contours else 0

def phanloai_kichthuoc(area):
    if area < 1000:
        return "nhỏ"
    elif area < 3000:
        return "vừa"
    return "lớn"

def chuyendoimau_hsv(images):
    return [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in images]

def trichxuatmau_hsv(hsv_images):
    return [{"mean_hue": np.mean(h), "mean_saturation": np.mean(s), "mean_value": np.mean(v)}
            for h, s, v in (cv2.split(img) for img in hsv_images)]

def tao_bangdulieu(features, labels, sizes):
    df = pd.DataFrame(features)
    df["label"] = labels
    df["size"] = sizes
    df["size_category"] = df["size"].apply(phanloai_kichthuoc)
    return df

def huanluyen_mohinh(df):
    X = df[["mean_hue", "mean_saturation", "mean_value", "size"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))
    return model

def dudoanketqua(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể tải ảnh từ {image_path}")
        return None, None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    size = tinh_kichthuoc(img)
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
        self.master.title("Kiểm tra trạng thái quả táo")
        tk.Label(self.master, text="Thư mục huấn luyện:").pack()
        tk.Button(self.master, text="Chọn thư mục", command=self.tai_thumuc_huanluyen).pack()
        tk.Label(self.master, text="Thư mục kiểm tra:").pack()
        tk.Button(self.master, text="Chọn thư mục kiểm tra", command=self.tai_thumuc_kiemtra).pack()
        tk.Button(self.master, text="Dự đoán", command=self.predict).pack()

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

        results = []
        for img_path in self.test_image_paths:
            label, size_category = dudoanketqua(img_path, self.model)
            results.append((img_path, label, size_category))

        self.show_results(results)

    def show_results(self, results):
        result_window = tk.Toplevel(self.master)
        result_window.title("Kết quả dự đoán")

        tree = ttk.Treeview(result_window, columns=("Image", "Label", "Size"), show="headings")
        tree.heading("Image", text="Hình ảnh")
        tree.heading("Label", text="Trạng thái")
        tree.heading("Size", text="Kích thước")
        tree.pack(fill=tk.BOTH, expand=True)

        for img_path, label, size in results:
            tree.insert("", tk.END, values=(img_path, label, size))

        def on_select(event):
            selected_item = tree.selection()
            if selected_item:
                item = tree.item(selected_item[0])
                img_path = item["values"][0]
                self.show_image(img_path)

        tree.bind("<<TreeviewSelect>>", on_select)

    def show_image(self, img_path):
        img_window = tk.Toplevel(self.master)
        img_window.title("Hình ảnh")

        img = Image.open(img_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)

        label = tk.Label(img_window, image=img_tk)
        label.image = img_tk
        label.pack()

# === Chạy ứng dụng ===
if __name__ == "__main__":
    root = tk.Tk()
    FruitClassifierApp(root)
    root.mainloop()
