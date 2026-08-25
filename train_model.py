import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# Use absolute paths to avoid FileNotFoundError
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DIRECTORY = os.path.join(BASE_PATH, "dataset")
CATEGORIES = ["with_mask", "without_mask"]

INIT_LR = 1e-4
EPOCHS = 20
BS = 32

print("[INFO] Loading images...")
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    if not os.path.exists(path):
        print(f"[ERROR] Path not found: {path}")
        continue
        
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        try:
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            data.append(image)
            labels.append(category)
        except Exception as e:
            print(f"[SKIP] Could not load image {img}: {e}")

if len(data) == 0:
    print("[ERROR] No images found! Check your dataset folders.")
    exit()

# Convert to arrays and encode labels
data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Data Augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, 
                         height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# Build Model (Transfer Learning with MobileNetV2)
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] Compiling and training...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

model.fit(aug.flow(trainX, trainY, batch_size=BS), 
          steps_per_epoch=len(trainX) // BS, 
          validation_data=(testX, testY), 
          validation_steps=len(testX) // BS, 
          epochs=EPOCHS)

# Save the model
model_path = os.path.join(BASE_PATH, "mask_detector.model")
model.save("mask_detector.h5")
print(f"[INFO] Model Saved successfully at: {model_path}")