from load_data import load_data
from model import create_model, train_model
from predict import predict_image
from tensorflow.keras.backend import clear_session

clear_session()

train, valid = load_data()
model=create_model()
train_model(model, train, valid)

print(predict_image('signs-to-predict/G.jpg', model, train))
print(predict_image('signs-to-predict/O.jpg', model, train))
