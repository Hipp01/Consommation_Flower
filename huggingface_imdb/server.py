import flwr as fl
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
import sys

csv_handler = CSVHandler('result_server.csv')
@measure_energy(handler=csv_handler)
def main():
    fl.server.start_server(server_address = 'localhost:'+str(sys.argv[1]), config={"num_rounds": 2})

if __name__ == "__main__":
    for i in range(100):
        main()
        csv_handler.save_data()
        print("\n\nTour : ",i,"\n\n")