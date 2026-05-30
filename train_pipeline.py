import os
import tensorflow as tf
import keras
from keras import layers
from keras.applications import MobileNetV2

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
DATASET_DIR = "dataset"
MODEL_NAME = "ecovision_material_model.keras"

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR, exist_ok=True)
    for cat in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]:
        os.makedirs(os.path.join(DATASET_DIR, cat), exist_ok=True)
    print(f"⚠ '{DATASET_DIR}' 구조가 초기화되었습니다. 각 폴더에 학습 데이터를 배치한 뒤 다시 가동하십시오.")
    exit()

train_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="int"
)
val_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="int"
)

class_names = train_ds.class_names
with open("class_names.txt", "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1)
])

base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = layers.Lambda(lambda img: tf.keras.applications.mobilenet_v2.preprocess_input(img))(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(MODEL_NAME, monitor="val_accuracy", save_best_only=True)
]

print("🚀 [Phase 1] 상위 분류 레이어 고속 동결 학습 개시...")
model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)

print("🚀 [Phase 2] 하부 레이어 일부 해제 기반 초정밀 미세조정(Fine-Tuning) 개시...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

print(f"🏆 경진대회 제출용 핵심 가중치 모델 파일 '{MODEL_NAME}' 마스터 빌드 완료.")
