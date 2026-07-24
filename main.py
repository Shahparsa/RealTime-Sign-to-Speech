#these are the libarires we used for the phase3

import cv2  
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import threading
import pyttsx3


#this is the exactly CNN part we implement in the phase 2 
class SignLanguageCNN(nn.Module):

    #here is the exatly definition of layers we have for the phase 2
    def __init__(self, num_classes=29):

        super(SignLanguageCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout2d_1 = nn.Dropout2d(p=0.10)
        self.dropout2d_2 = nn.Dropout2d(p=0.15)
        self.dropout2d_3 = nn.Dropout2d(p=0.20)
        self.dropout2d_4 = nn.Dropout2d(p=0.25)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(p=0.50)
        self.fc2 = nn.Linear(256, num_classes)


    #here is the exact forward part we have for phase 2 in our CNN 
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d_1(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d_2(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d_3(x)

        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2d_4(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x

#in this method we just use the engine to convert the text to the voice
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


#we exit the program if we press q or click the exit with mouse
#here we handle the exit with the mouse
exit_app = False
def mouse_click(event, x, y, flags, param):
    global exit_app
    if event == cv2.EVENT_LBUTTONDOWN:
        if 500 <= x <= 600 and 20 <= y <= 60:
            exit_app = True

def main():
    #in this part we define the classes and thier indexes . 
    global exit_app
    exit_app = False
    
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               'del', 'nothing', 'space'] 
    
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}

    #here if we have cuda libraries use GPU if we dont use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignLanguageCNN(num_classes=len(classes)).to(device)
    
    ##this is the part witch we use our weights that we calculated in the prevoius phase and trained on our dataset
    model.load_state_dict(torch.load('sign_language_cnn_model.pth', map_location=device))
    model.eval()
    print("Trained model loaded successfully!")

    #this is the String we build from predicted charachters
    current_text = ""
    #here we store the last predicted Class
    last_predicted_class = None
    #we start this when the stable sign recognized and we want to wait a little time 
    class_start_time = time.time()
    #we use this flag for not duplicating the Signs /
    char_added_flag = False
    #and we want 5 second when nothing sign comes so we need timer for this too
    nothing_start_time = None
    #if our prediction certain probability is above below number then we assume it as certain
    CONFIDENCE_THRESHOLD = 0.70 
    
    #here is the rectangle for our hand in webcam such that our model just see the Fram there for predicting
    x_min, y_min, x_max, y_max = 50, 50, 300, 300

    #here is the camera setup and the app window prepration 
    cap = cv2.VideoCapture(0)
    print("Webcam started. Click the 'EXIT' button or press 'q' to exit.")
    window_name = 'Sign Language Live Recognition'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)

    while True:
        #in the loop we get the Frame . and if Frame doesnt Exist we log the error for the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        display_frame = frame.copy()
        frame_time = time.time()

        #this tow lines create the box of hand and the Exit button    
        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.rectangle(display_frame, (500, 20), (600, 60), (0, 0, 255), -1)
        cv2.putText(display_frame, 'EXIT', (520, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        #here we Crop the hand_box part from the Frame to pass it to our model
        roi = frame[y_min:y_max, x_min:x_max]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        resized_roi = cv2.resize(rgb_roi, (64, 64))
        normalized_roi = resized_roi.astype(np.float32) / 255.0

        #this 2 lines just convert our picture data in to the tensors for the Torch library        
        tensor_roi = torch.tensor(normalized_roi).permute(2, 0, 1)
        input_tensor = tensor_roi.unsqueeze(0).to(device)


        # here it procces the CNN and now sww witch class is predicted
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            max_prob, predicted_idx = probs.max(1)
            
        confidence = max_prob.item()
        predicted_class = idx_to_class[predicted_idx.item()]

        #if we are certain that the answer is the correct one
        if confidence > CONFIDENCE_THRESHOLD:
            #if the class is Nothing and we wait 5 second it speak the string 
            if predicted_class == 'nothing':
                if nothing_start_time is None:
                    nothing_start_time = frame_time
                elif frame_time - nothing_start_time >= 5.0:
                    if len(current_text.strip()) > 0:
                        threading.Thread(target=speak_text, args=(current_text,)).start()
                        current_text = "" 
                    nothing_start_time = frame_time 
            else:
                nothing_start_time = None 
            
            #here when we want to add the letter when it stayes at least 2 seconds . 
            if predicted_class == last_predicted_class:
                if frame_time - class_start_time >= 2.0:
                    if not char_added_flag: 
                        if predicted_class == 'space':
                            current_text += "-"
                        elif predicted_class == 'del':
                            current_text = current_text[:-1]
                        elif predicted_class != 'nothing':
                            current_text += predicted_class
                        
                        char_added_flag = True
            #here when the predicted_class is new and we update the time , flag and the last_predicted class . 
            else:
                last_predicted_class = predicted_class
                class_start_time = frame_time
                char_added_flag = False
                
        else:
            predicted_class = "Uncertain"

        #here we print the predicted class and the text and other graphical stuffs
        cv2.putText(display_frame, f'Pred: {predicted_class} ({confidence*100:.1f}%)', (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255), 2)
        cv2.putText(display_frame, f'Text: {current_text}', (20, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        cv2.imshow(window_name, display_frame)
        if exit_app or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()