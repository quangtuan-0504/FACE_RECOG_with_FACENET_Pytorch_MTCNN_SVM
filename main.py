import cv2
from facenet_pytorch import MTCNN,InceptionResnetV1
import time
import numpy as np
import torch
import joblib

COLOR_BOX = (0, 0, 255)
COLOR_TEXT = (0, 0, 255)
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
SIZE_TEXT = 1

mtcnn = MTCNN(keep_all=False)
face_vectorizer_resnet = InceptionResnetV1(pretrained='vggface2').eval()
# Load mô hình từ tệp tin đã lưu
loaded_model_SVM = joblib.load('svm_model.pkl')
#Load label enocder
label_encoder_load = joblib.load("label_encoder.pkl")


cap = cv2.VideoCapture(0)


def _draw(frame, boxes, probs, landmarks ,name = None):
    """
    Draw landmarks and boxes for each face detected
    """
    if boxes is None:
        return frame
    for box, prob, ld in zip(boxes, probs, landmarks):# vẽ từng box
        box = list(map(lambda x: int(x), box))

        # Draw rectangle on frame
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), COLOR_BOX, THICKNESS)
        if name is not None:
            cv2.putText(frame, name, (box[0], box[1]-20), FONT, SIZE_TEXT, COLOR_TEXT, THICKNESS, cv2.LINE_AA)
        # Write text
        else:
            cv2.putText(frame, "Face score " + str(round(prob,2)), (box[0], box[1]-20), FONT, SIZE_TEXT, COLOR_TEXT, THICKNESS, cv2.LINE_AA)



    return frame


while True:

            name = None
            start = time.time()
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            try:
                # detect face box, probability and landmarks
                boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)

                #check format ouput mtcnn
                #print(boxes) ko có face thì None
                #print(type(boxes))
                #print(boxes.shape)


                if boxes is not None:

                    #crop_face
                    box=boxes[0]
                    x1,y1,x2,y2=box.astype(int)
                    face_croped=frame[y1:y2,x1:x2]
                    face_croped=cv2.resize(face_croped,(160,160))
                    #print(face_croped.shape)

                    #convert2torch tensor
                    face_croped_torch= torch.from_numpy(face_croped).permute(2, 0, 1).float().div(255.0)
                    #print((face_croped_torch.shape))
                    # Calculate embedding (unsqueeze to add batch dimension)
                    img_embedding = face_vectorizer_resnet(face_croped_torch.unsqueeze(0)).detach().cpu()# unsqueeze là để thêm ngoặc bên ngoài
                    #print(img_embedding.shape)



                    # Sử dụng mô hình để dự đoán
                    result = loaded_model_SVM.predict_proba(img_embedding)
                    #print(result)

                    # select argmax probability
                    max_proba = np.max(result)
                    idx_max = np.argmax(result)
                    print(max_proba,idx_max)
                    if max_proba >= 0.7:

                        #Invert
                        Name_recog = label_encoder_load.inverse_transform([idx_max])
                        name = Name_recog[0] + " cls score " + str(round(max_proba,2))
                        # print(Name_recog)
                # draw on frame
                _draw(frame, boxes, probs, landmarks ,name)
            except:
                print('THIS FRAME IS FAILURE!!!')

            # Show the frame
            cv2.imshow('Face Detection', frame)
            print('FPS: %.2f' % (1 / (time.time() - start)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()