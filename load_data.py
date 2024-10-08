from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    data = 'data'
    
    datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2
)

    # training set
    train = datagen.flow_from_directory(
        data,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        seed=123
    )

    # validation set
    valid = datagen.flow_from_directory(
        data,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        seed=123
    )
    return train, valid