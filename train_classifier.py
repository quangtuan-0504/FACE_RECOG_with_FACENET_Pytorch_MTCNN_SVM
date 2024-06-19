import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer
import joblib

# Định nghĩa đường dẫn tới tập dữ liệu
data_dir = 'DATABASE_FACE/IMG'

# Sử dụng MTCNN để phát hiện khuôn mặt
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# Sử dụng InceptionResnetV1 để trích xuất đặc trưng khuôn mặt
resnet = InceptionResnetV1(pretrained='vggface2').eval()


# Hàm để lấy tất cả ảnh và nhãn từ thư mục
def get_images_and_labels(data_dir):
    images = []
    labels = []
    label_names = sorted(os.listdir(data_dir))

    for label in label_names:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):# nếu thư mục này tồn tại
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                if image_path.endswith(('.jpg', '.jpeg', '.png')):
                    images.append(image_path)
                    labels.append(label)

    return images, labels, label_names


# Lấy tất cả ảnh và nhãn từ thư mục
image_paths, labels, label_names = get_images_and_labels(data_dir)

# Mã hóa nhãn
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Lưu tên các nhãn đã mã hóa để sử dụng sau
joblib.dump(label_encoder, 'label_encoder.pkl')

# Danh sách để lưu trữ các vector nhúng và nhãn
embeddings = []
encoded_labels = []


# Hàm để trích xuất vector nhúng từ ảnh
def extract_embeddings(image_paths, labels_encoded):
    for img_path, label in zip(image_paths, labels_encoded):
        #print(img_path)
        img = Image.open(img_path).convert('RGB')
        #print(img.size)
        img_cropped = mtcnn(img)
        #chỗ này nó trả về 2 face vì ở trên mình đặt keep_all=True, file test để False nên nó chỉ trả về 1 face.
        #print(img_cropped.shape)
        #break

        if img_cropped is not None:
            # Chỉ lấy khuôn mặt đầu tiên được phát hiện
            img_embedding = resnet(img_cropped[0].unsqueeze(0)).detach().cpu()
            embeddings.append(img_embedding.numpy())
            encoded_labels.append(label)


# Trích xuất vector nhúng từ tất cả các ảnh
extract_embeddings(image_paths, labels_encoded)

# Chuyển danh sách thành mảng numpy
embeddings = np.array(embeddings).squeeze()  # (num_samples, 512)
encoded_labels = np.array(encoded_labels)

# Chuẩn hóa vector nhúng
in_encoder = Normalizer(norm='l2')
embeddings = in_encoder.transform(embeddings)

# Huấn luyện SVM
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(embeddings, encoded_labels)

# Lưu mô hình SVM
joblib.dump(svm_model, 'svm_model.pkl')

print("Hoàn thành việc trích xuất vector nhúng và huấn luyện SVM.")
