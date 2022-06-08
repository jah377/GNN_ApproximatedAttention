
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_slices(dim_size, batch_size):
    """ create generator of index slices """

    count = 0
    while True:
        yield slice(count, count + int(batch_size), 1)
        count += int(batch_size)
        if count >= int(dim_size):
            break


def cosine_filter(x, edge_index, args):
    """ create cosine similiarity attention filter """

    assert hasattr(args, 'CS_BATCH_SIZE'), 'Must specify CS_BATCH_SIZE value'

    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]

    values = torch.tensor(range(num_edges)).cpu()

    for batch in create_slices(num_edges, args.CS_BATCH_SIZE):
        A = x[edge_index[0, batch]].to(device)
        B = x[edge_index[1, batch]].to(device)
        values[batch] = F.cosine_similarity(A, B, dim=1).cpu()

        del A, B
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=F.relu(values),  # normalize [0-1]
        sparse_sizes=(num_nodes, num_nodes)
    )
