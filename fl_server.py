import flwr as fl
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, Parameters, Scalar

# Custom Strategy Class for better logging
class CustomStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def aggregate_evaluate(self, rnd, results, failures):
        loss, metrics = super().aggregate_evaluate(rnd, results, failures)
        
        print(f"\n[Server] Round {rnd} Evaluation Results:")
        print(f"  ➤ Aggregated Loss   : {loss}")
        print(f"  ➤ Aggregated Accuracy: {metrics.get('accuracy', 'N/A')}\n")
        
        return loss, metrics

# Correct signature for metrics aggregation function
def aggregate_evaluate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics from multiple clients."""
    if not metrics:
        return {"accuracy": 0.0}
    
    # Extract accuracies from client metrics
    accuracies = [m["accuracy"] for _, m in metrics]
    
    # Calculate average accuracy
    avg_accuracy = sum(accuracies) / len(accuracies)
    
    return {"accuracy": avg_accuracy}

if __name__ == "__main__":
    port = 8082  # Using a different port
    server_address = f"localhost:{port}"
    print(f"Starting Flower server on {server_address}")
    
    # Create strategy with consistent parameters
    strategy = CustomStrategy(
        fraction_fit=0.5,  # Sample 50% of clients for training
        min_fit_clients=7,  # Require at least 2 clients for training
        min_available_clients=7,  # Wait for at least 2 clients before starting
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics  # Fixed function
    )
    
    # Start server with proper configuration
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy
    )