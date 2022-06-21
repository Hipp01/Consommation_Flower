import flwr as fl
import tensorflow as tf
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
import sys

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=5, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


csv_handler = CSVHandler('result_client_epochs_5.csv')
@measure_energy(handler=csv_handler)
def main():
    fl.client.start_numpy_client(server_address="localhost:"+str(sys.argv[1]), client=CifarClient())


if __name__ == "__main__":
    main()
    csv_handler.save_data()
