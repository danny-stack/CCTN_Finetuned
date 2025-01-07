# client.py
import tkinter as tk
from tkinter import filedialog
import requests
import base64
from PIL import Image, ImageTk
import numpy as np
import cv2

class TableDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Table Detection")
        
        # 选择图片按钮
        self.select_btn = tk.Button(root, text="选择图片", command=self.select_image)
        self.select_btn.pack(pady=10)
        
        # 显示图片
        self.image_label = tk.Label(root)
        self.image_label.pack()
        
        # 检测按钮
        self.detect_btn = tk.Button(root, text="检测表格", command=self.detect_tables)
        self.detect_btn.pack(pady=10)
        
        # 结果显示
        self.result_text = tk.Text(root, height=10)
        self.result_text.pack(pady=10)
        
        self.image_path = None
        
    def select_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if self.image_path:
            # 显示选择的图片
            img = Image.open(self.image_path)
            img.thumbnail((800, 800))  # 缩放显示
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo)
            self.image_label.image = photo


    def show_image(self, image_path, boxes=None):
        img = cv2.imread(image_path)
        orig_h, orig_w = img.shape[:2]
        scale_w = orig_w / 1333
        scale_h = orig_h / 800
        
        if boxes:
            for box in boxes:
                x = int(box['bbox']['left'] * scale_w)
                y = int(box['bbox']['top'] * scale_h)
                w = int(box['bbox']['width'] * scale_w)
                h = int(box['bbox']['height'] * scale_h)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
    
    # def detect_tables(self):
    #     if not self.image_path:
    #         return
            
    #     # 读取图片并转换为base64
    #     with open(self.image_path, "rb") as img_file:
    #         img_base64 = base64.b64encode(img_file.read()).decode()
            
    #     # 发送请求到API
    #     try:
    #         response = requests.post(
    #             "http://localhost:6006/detect_tables",
    #             json={"image_base64": img_base64}
    #         )
    #         result = response.json()
    #         self.result_text.delete(1.0, tk.END)
    #         self.result_text.insert(tk.END, str(result))
    #     except Exception as e:
    #         self.result_text.delete(1.0, tk.END)
    #         self.result_text.insert(tk.END, f"Error: {str(e)}")

    def detect_tables(self):
        if not self.image_path:
            return
        with open(self.image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
            
        try:
            response = requests.post(
                "http://localhost:6006/detect_tables",
                json={"image_base64": img_base64}
            )
            boxes = response.json()
            self.show_image(self.image_path, boxes)
        except Exception as e:
            tk.messagebox.showerror("Error", str(e))

    # def show_metrics(self):
    #     try:
    #         response = requests.get("http://localhost:6006/metrics")
    #         metrics = response.json()
    #         tk.messagebox.showinfo("评估指标", 
    #             f"Precision: {metrics['precision']:.3f}\n"
    #             f"Recall: {metrics['recall']:.3f}\n"
    #             f"F1-score: {metrics['f1_score']:.3f}")
    #     except Exception as e:
    #         tk.messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = TableDetectionApp(root)
    root.mainloop()