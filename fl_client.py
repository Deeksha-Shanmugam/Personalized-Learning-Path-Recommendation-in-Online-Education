import flwr as fl
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import multiprocessing
import time
from model import Net  # Import your model
from create_clients import create_clients  # Import your data creation function

# Client function
def client_fn(cid, clients_data, features):
    # Get the correct client ID and data
    cid = int(cid)
    client_id = list(clients_data.keys())[cid]
    print(f"Starting client {cid} with ID: {client_id}")
    
    # Prepare data
    X, y = clients_data[client_id]
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize model and optimizer
    model = Net(features)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Define Flower client
    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().detach().numpy() for val in model.parameters()]
            
        def set_parameters(self, parameters):
            for param, val in zip(model.parameters(), parameters):
                param.data = torch.tensor(val, dtype=torch.float32)
                
        def fit(self, parameters, config):
            self.set_parameters(parameters)
            model.train()
            total_loss = 0.0
            
            # Add this:
            local_epochs = 10  # Train 5 times over the dataset

            for epoch in range(local_epochs):
                for X_batch, y_batch in train_loader:
                    pred = model(X_batch)
                    loss = loss_fn(pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            print(f"Client {client_id} completed training round")
            return self.get_parameters(config), len(train_loader.dataset), {"loss": total_loss / (len(train_loader) * local_epochs)}
        
        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            model.eval()
            correct, total = 0, 0
            total_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in train_loader:
                    outputs = model(X_batch)
                    loss = loss_fn(outputs, y_batch)
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)
                    
            accuracy = correct / total
            print(f"Client {client_id} evaluation - Accuracy: {accuracy:.4f}")
            return total_loss / total, total, {"accuracy": accuracy}
    
    # Create client and convert to standard Client (fixes deprecation warning)
    client = FlowerClient().to_client()
    
    # Start client
    fl.client.start_client(server_address="localhost:8082", client=client)

# Function to start client processes
def start_client_process(cid, clients_data, features):
    client_fn(cid, clients_data, features)

if __name__ == "__main__":
    # Load client data
    #clients_data = create_clients()
    clients_data = create_clients(num_clients=7)

    features = list(clients_data.values())[0][0].shape[1]
    
    # Number of clients to run
    num_clients = 7
    
    # Start clients in separate processes
    processes = []
    for i in range(num_clients):
        p = multiprocessing.Process(
            target=start_client_process,
            args=(i, clients_data, features)
        )
        p.start()
        processes.append(p)
        # Slight delay to avoid race conditions
        time.sleep(0.5)
    
    # Wait for all clients to complete
    for p in processes:
        p.join()