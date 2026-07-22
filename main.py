import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import threading
import pyttsx3

# ۱. تعریف مجدد معماری شبکه
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=29):
        super(SignLanguageCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# تابع برای اجرای TTS در یک Thread جداگانه تا وبکم هنگ نکند
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    # ۲. تعریف لیست کلاس‌ها
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               'del', 'nothing', 'space'] 
    
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}

    # ۳. آماده‌سازی دستگاه و بارگذاری مدل
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignLanguageCNN(num_classes=len(classes)).to(device)
    
    # اطمینان حاصل کنید که مسیر فایل مدل درست است
    model.load_state_dict(torch.load('sign_language_cnn_model.pth', map_location=device))
    model.eval()
    print("Trained model loaded successfully!")

    # متغیرهای مربوط به منطق پیوسته
    current_text = ""
    last_predicted_class = None
    class_start_time = time.time()
    char_added_flag = False
    nothing_start_time = None
    CONFIDENCE_THRESHOLD = 0.7 # آستانه اطمینان بالا
    
    # مختصات کادر دست (Bounding Box)
    x_min, y_min, x_max, y_max = 50, 50, 300, 300

    # ۴. راه‌اندازی وبکم
    cap = cv2.VideoCapture(0)
    print("Webcam started. Press 'q' on the video window to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        display_frame = frame.copy()
        frame_time = time.time()

        # رسم کادر روی تصویر نمایشی
        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # استخراج محتوای داخل کادر برای پیش‌پردازش و دادن به مدل
        roi = frame[y_min:y_max, x_min:x_max]

        # پیش‌پردازش فریم (فقط بخش ROI)
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        resized_roi = cv2.resize(rgb_roi, (64, 64))
        normalized_roi = resized_roi.astype(np.float32) / 255.0
        
        tensor_roi = torch.tensor(normalized_roi).permute(2, 0, 1)
        input_tensor = tensor_roi.unsqueeze(0).to(device)

        # پیش‌بینی
        with torch.no_grad():
            outputs = model(input_tensor)
            # استفاده از Softmax برای به دست آوردن میزان اطمینان (Confidence)
            probs = F.softmax(outputs, dim=1)
            max_prob, predicted_idx = probs.max(1)
            
        confidence = max_prob.item()
        predicted_class = idx_to_class[predicted_idx.item()]

        # بررسی اطمینان بالا برای اعمال منطق
        if confidence > CONFIDENCE_THRESHOLD:
            
            # ----------------------------------------------------
            # منطق (ب): ۵ ثانیه کادر خالی (Nothing) برای نهایی کردن رشته
            # ----------------------------------------------------
            if predicted_class == 'nothing':
                if nothing_start_time is None:
                    nothing_start_time = frame_time
                elif frame_time - nothing_start_time >= 5.0:
                    if len(current_text.strip()) > 0:
                        # پخش صدای رشته متنی
                        threading.Thread(target=speak_text, args=(current_text,)).start()
                        current_text = "" # پاک کردن رشته برای کلمات بعدی
                    nothing_start_time = frame_time # ریست کردن زمان برای جلوگیری از پخش تکراری
            else:
                nothing_start_time = None # اگر دست وارد کادر شد، تایمر nothing صفر می‌شود
            
            # ----------------------------------------------------
            # منطق (آ): ۲ ثانیه متوالی برای افزودن یک حرف
            # ----------------------------------------------------
            if predicted_class == last_predicted_class:
                if frame_time - class_start_time >= 2.0:
                    if not char_added_flag: # بررسی اینکه حرف تکراری چندبار ثبت نشود
                        if predicted_class == 'space':
                            current_text += " "
                        elif predicted_class == 'del':
                            current_text = current_text[:-1]
                        elif predicted_class != 'nothing':
                            current_text += predicted_class
                        
                        char_added_flag = True # پرچم روشن می‌شود تا وقتی کلاس عوض نشد حرفی اضافه نشود
            else:
                last_predicted_class = predicted_class
                class_start_time = frame_time
                char_added_flag = False
                
        else:
            predicted_class = "Uncertain"

        # ----------------------------------------------------
        # نمایش اطلاعات روی صفحه
        # ----------------------------------------------------
        cv2.putText(display_frame, f'Current Pred: {predicted_class} ({confidence:.2f})', (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(display_frame, f'Final Text: {current_text}', (20, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow('Sign Language Live Recognition', display_frame)

        # خروج با کلید q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()