import pandas as pd
import numpy as np

# Load Dataset

df = pd.read_csv("dataset/fer2013.csv")

X = []

for pixels in df["pixels"]:
    image = np.array(pixels.split(), dtype="float32").reshape(48, 48)
    X.append(image)

X = np.array(X)

y = df["emotion"].values

print(f"Image Shape : {X.shape}")
print(f"Label Shape : {y.shape}")


# Preprocessing

X = X / 255.0
X = X.reshape(-1, 48, 48, 1)

print(f"Processed Image Shape : {X.shape}")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# One Hot Encoding

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

print(f"Training Images : {X_train.shape}")
print(f"Testing Images  : {X_test.shape}")

# ==========================================
# Build CNN Model
# ==========================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Flatten,Dense,Dropout

model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu", input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(256, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))


model.add(Dense(7, activation="softmax"))

model.summary()

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

# Callbacks

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

early_stop = EarlyStopping(monitor="val_loss",patience=7,restore_best_weights=True)

checkpoint = ModelCheckpoint("models/emotion_model.keras",monitor="val_accuracy",save_best_only=True,mode="max",verbose=1)

reduce_lr = ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=3,min_lr=1e-6,verbose=1)

# Train Model

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop,checkpoint,reduce_lr])

# Evaluate Model

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

# Check Overfitting / Underfitting

print("\nTraining Performance")
print(f"Training Accuracy   : {history.history['accuracy'][-1]:.4f}")
print(f"Validation Accuracy : {history.history['val_accuracy'][-1]:.4f}")

print(f"Training Loss       : {history.history['loss'][-1]:.4f}")
print(f"Validation Loss     : {history.history['val_loss'][-1]:.4f}")

# Save Model

model.save("models/emotion_model4.keras")

print("\nModel saved successfully!")