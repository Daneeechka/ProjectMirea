import tensorflow as tf
from tensorflow.keras.applications import ResNet50  # Или другая подходящая архитектура
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Загрузка предобученной модели (ResNet50 - пример)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 2. Добавление собственных слоев
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Можно изменить количество нейронов
predictions = Dense(1, activation='sigmoid')(x)  # 1 - если есть машина, 0 - нет машины

# 3. Создание модели
model = Model(inputs=base_model.input, outputs=predictions)

# 4. Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Генерация данных (ImageDataGenerator)  -  важно для аугментации
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% данных для валидации
)

train_generator = train_datagen.flow_from_directory(
    'C:/Users/gia/Downloads/archive',  # <---  Путь к тренировочному датасету
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # 'binary' если одна машина или нет машины на изображении
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'C:/Users/gia/Downloads/archive',  # <---  Путь к тренировочному датасету
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 6. Обучение модели
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,  # Количество эпох -  надо подбирать
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# 7. Сохранение модели
model.save('parking_car_detector.h5')

img = 
