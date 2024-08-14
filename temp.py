import torch
from torch_geometric.data import Data

# Minimaler Testfall
def test_minimal_data_creation():
    # Beispielhafte Knotendaten (x)
    x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)

    # Beispielhafte Kanteninformationen (edge_index)
    edge_index = torch.tensor([[0, 1, 2],
                               [1, 2, 0]], dtype=torch.long)

    # Beispielhafte Kantendaten (edge_attr)
    edge_attr = torch.tensor([[0.1, 0.2],
                              [0.3, 0.4],
                              [0.5, 0.6]], dtype=torch.float)

    # Beispielhafte Zielvariable (y)
    y = torch.tensor([1, 0, 1], dtype=torch.float).view(-1, 1)

    # Versuch, ein einfaches Data-Objekt zu erstellen
    try:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        print("Successfully created Data object:", data)
    except Exception as e:
        print("Failed to create Data object:", e)

test_minimal_data_creation()
