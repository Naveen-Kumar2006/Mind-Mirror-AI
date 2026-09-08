import pandas as pd
import numpy as np

# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv(r"C:\projects\MindMirror AI\data\fer2013.csv")

X = []

for pixels in df["pixels"]:
    image = np.array(
        pixels.split(),
        dtype="float32"
    ).reshape(48, 48)

    X.append(image)

X = np.array(X)
y = df["emotion"].values

print(f"Image Shape : {X.shape}")
print(f"Label Shape : {y.shape}")


# ==========================================
# 2. Preprocessing
# ==========================================

# Normalize pixel values
X = X / 255.0

# Add grayscale channel
X = X.reshape(-1, 48, 48, 1)

print(f"Processed Image Shape : {X.shape}")


# ==========================================
# 3. Train / Validation / Test Split
# ==========================================

from sklearn.model_selection import train_test_split

# First split:
# 90% → Train + Validation
# 10% → Test

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42,
    stratify=y
)

# Second split:
# From the remaining 90%:
# 10% → Validation
# 80% → Train

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.1111,
    random_state=42,
    stratify=y_temp
)

print("\nDataset Split:")
print(f"Training Images   : {X_train.shape}")
print(f"Validation Images : {X_val.shape}")
print(f"Testing Images    : {X_test.shape}")


# ==========================================
# 4. One Hot Encoding
# ==========================================

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=7)
y_val = to_categorical(y_val, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)


# ==========================================
# 5. Build CNN Model
# ==========================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D,MaxPooling2D,BatchNormalization,Flatten,Dense,Dropout)

model = Sequential()

# Block 1
model.add(Conv2D(32,(3, 3),activation="relu",input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))


# Block 2
model.add(Conv2D(64,(3, 3),activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))


# Block 3
model.add(Conv2D( 128,(3, 3),activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))


# Block 4
model.add(Conv2D(256,(3, 3),activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))


# Fully Connected Layers
model.add(Flatten())

model.add(Dense(256,activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(128,activation="relu"))

model.add(Dropout(0.3))


# Output Layer
model.add(Dense(7,activation="softmax" ))


# ==========================================
# 6. Model Summary
# ==========================================

model.summary()


# ==========================================
# 7. Compile Model
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 8. Callbacks
# ==========================================

from tensorflow.keras.callbacks import (EarlyStopping,ModelCheckpoint,ReduceLROnPlateau)

early_stop = EarlyStopping(monitor="val_loss",patience=7,restore_best_weights=True,verbose=1)

checkpoint = ModelCheckpoint("models/emotion_model.keras",monitor="val_accuracy",save_best_only=True,mode="max",verbose=1)

reduce_lr = ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=3,min_lr=1e-6,verbose=1)


# ==========================================
# 9. Train Model
# ==========================================

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=50,
    batch_size=64,

    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)


# ==========================================
# 10. Load Best Model
# ==========================================

from tensorflow.keras.models import load_model


best_model = load_model(r"models/emotion_model.keras")


# ==========================================
# 11. Final Evaluation on TEST Dataset
# ==========================================

test_loss, test_accuracy = best_model.evaluate(X_test,y_test,verbose=1)

print("\n==========================================")
print("FINAL TEST RESULTS")
print("==========================================")

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")


# ==========================================
# 12. Training / Validation Performance
# ==========================================

print("\n==========================================")
print("TRAINING RESULTS")
print("==========================================")

print(f"Best Validation Accuracy : "f"{max(history.history['val_accuracy']):.4f}")

print(f"Best Validation Loss     : "f"{min(history.history['val_loss']):.4f}")


# ==========================================
# 13. Save Final Model
# ==========================================

best_model.save(r"models/emotion_model_final.keras")

print("\nModel saved successfully!")

print("\nFiles created:")
print("models/emotion_model.keras")
print("models/emotion_model_final.keras")