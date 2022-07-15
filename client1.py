import flwr as fl
import tensorflow as tf
import sys
from tensorflow import keras
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

traingen = ImageDataGenerator(rescale= 1./255,
                             width_shift_range=0.2 , 
                             height_shift_range=0.2 ,
                             zoom_range=0.2)
valgen = ImageDataGenerator(rescale= 1./255)
testgen = ImageDataGenerator(rescale= 1./255)

train_it = traingen.flow_from_directory("D:/Federated/client1_data/train", target_size = (224, 224))
val_it = traingen.flow_from_directory("D:/Federated/client1_data/val", target_size = (224, 224))
test_it = traingen.flow_from_directory("D:/Federated/client1_data/test", target_size = (224, 224))


base_model_201 = tf.keras.applications.DenseNet201(input_shape = (224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# lock layers
for layer in base_model_201.layers:
  layer.trainable = False

x = layers.Flatten()(base_model_201.output)  # base_model_201.output
x = layers.Dropout(0.5)(x) # 
x = layers.Dense(512, activation= 'relu')(x)
x = layers.Dense(5, activation = 'softmax')(x)

model = tf.keras.models.Model(base_model_201.input, x) 

# compile
model.compile('adam', loss = 'categorical_crossentropy',metrics = ['acc'])



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(train_it, validation_data= val_it, epochs=1, steps_per_epoch=60, validation_steps=10)
        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), len(train_it), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_it, steps= 1)
        print("Eval accuracy : ", accuracy)
        return loss, len(test_it), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)