import flwr as fl
import sys
import numpy as np

from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

# Create strategy and run server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
csv_handler = CSVHandler('result_server.csv')
@measure_energy(handler=csv_handler)
def main():
    
    fl.server.start_server(
            server_address = 'localhost:'+str(sys.argv[1]) , 
            config={"num_rounds": 2} ,
            grpc_max_message_length = 1024*1024*1024,
            strategy = strategy
    )



if __name__ == "__main__":
    for i in range(300):
        main()
        csv_handler.save_data()
        print("Tour : ",i)