import time
import dill as pickle
import numpy as np
import itertools
from scipy.spatial.distance import pdist, squareform
import torch
from torch.utils.data import Dataset
import dgl
from packaging import version

class TSP(Dataset):
    def __init__(self, data_dir, split="train", num_neighbors=25, max_samples=10000):    
        self.data_dir = data_dir
        self.split = split
        self.filename = f'{data_dir}/tsp50-500_{split}.txt'
        self.max_samples = max_samples
        self.num_neighbors = num_neighbors
        self.is_test = split.lower() in ['test', 'val']
        
        self.graph_lists = []
        self.edge_labels = []
        self.n_samples = 0  # Initialize sample count
        
        # Detect DGL version
        self.dgl_version = getattr(dgl, '__version__', '0.4.0')
        print(f"DGL version detected: {self.dgl_version}")
        
        self._prepare()
        self.n_samples = len(self.edge_labels)
        
        # Save the graphs after preparation
        self._save_graphs()

    def _prepare(self):
        print(f'Preparing all graphs for the {self.split.upper()} set...')
        
        file_data = open(self.filename, "r").readlines()[:self.max_samples]
        
        for graph_idx, line in enumerate(file_data):
            line = line.strip().split(" ")  # Split into list and remove any extra whitespace
            num_nodes = int(line.index('output') // 2)
            
            # Convert node coordinates to required format
            nodes_coord = []
            for idx in range(0, 2 * num_nodes, 2):
                nodes_coord.append([float(line[idx]), float(line[idx + 1])])

            # Compute distance matrix
            W_val = squareform(pdist(nodes_coord, metric='euclidean'))
            # Determine k-nearest neighbors for each node
            knns = np.argpartition(W_val, kth=self.num_neighbors, axis=-1)[:, :self.num_neighbors+1]

            # Convert tour nodes to required format
            # Don't add final connection for tour/cycle
            output_idx = line.index('output')
            tour_nodes = [int(node) - 1 for node in line[output_idx + 1:]][:-1]

            # Compute an edge adjacency matrix representation of tour
            edges_target = np.zeros((num_nodes, num_nodes))
            for idx in range(len(tour_nodes) - 1):
                i = tour_nodes[idx]
                j = tour_nodes[idx + 1]
                edges_target[i][j] = 1
                edges_target[j][i] = 1
            # Add final connection of tour in edge target
            edges_target[tour_nodes[-1]][tour_nodes[0]] = 1
            edges_target[tour_nodes[0]][tour_nodes[-1]] = 1
            
            # Construct the DGL graph
            if version.parse(self.dgl_version) < version.parse("0.5.0"):
                # For older versions of DGL
                g = dgl.DGLGraph()
                g.add_nodes(num_nodes)
            else:
                # For newer versions of DGL
                g = dgl.graph(([], []), num_nodes=num_nodes)
            
            g.ndata['feat'] = torch.tensor(nodes_coord, dtype=torch.float32)
            
            edge_feats = []  # Edge features (Euclidean distances)
            edge_labels = []  # Edge labels (1 if part of the tour, 0 otherwise)
            src_nodes = []
            dst_nodes = []
            for idx in range(num_nodes):
                for n_idx in knns[idx]:
                    if n_idx != idx:  # No self-connections
                        src_nodes.append(idx)
                        dst_nodes.append(n_idx)
                        edge_feats.append(W_val[idx][n_idx])
                        edge_labels.append(int(edges_target[idx][n_idx]))
            
            # Add edges to the graph
            if version.parse(self.dgl_version) < version.parse("0.5.0"):
                # For older versions of DGL
                g.add_edges(src_nodes, dst_nodes)
            else:
                # For newer versions of DGL
                g.add_edges(src_nodes, dst_nodes)
            
            # Add edge features
            g.edata['feat'] = torch.tensor(edge_feats, dtype=torch.float32).unsqueeze(-1)
            g.edata['label'] = torch.tensor(edge_labels, dtype=torch.int64)
            
            self.graph_lists.append(g)
            self.edge_labels.append(edge_labels)

    def _save_graphs(self):
        # Save the graphs
        graph_path = f'{self.data_dir}/{self.split}_graphs.bin'
        label_path = f'{self.data_dir}/{self.split}_labels.pkl'
        
        # For newer versions, labels can be saved within the graphs
        if version.parse(self.dgl_version) >= version.parse("0.5.0"):
            dgl.save_graphs(graph_path, self.graph_lists)
            print(f"Graphs saved to {graph_path}")
            # If needed, save labels separately
            with open(label_path, 'wb') as f:
                pickle.dump(self.edge_labels, f)
                print(f"Labels saved to {label_path}")
        else:
            # For older versions, save graphs and labels separately
            dgl.data.utils.save_graphs(graph_path, self.graph_lists)
            print(f"Graphs saved to {graph_path}")
            with open(label_path, 'wb') as f:
                pickle.dump(self.edge_labels, f)
                print(f"Labels saved to {label_path}")

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get the idx^th sample.
        Returns
        -------
        (dgl.DGLGraph, list)
            DGLGraph with node features in 'feat' field and edge labels.
        """
        g = self.graph_lists[idx]

        # Initialize dgl_version if it does not exist
        if not hasattr(self, 'dgl_version'):
            self.dgl_version = getattr(dgl, '__version__', '0.4.0')

        # For newer versions, extract labels from edata if needed
        if version.parse(self.dgl_version) >= version.parse("0.5.0"):
            edge_labels = g.edata['label'].tolist()
        else:
            edge_labels = self.edge_labels[idx]
        return g, edge_labels

class TSPDatasetDGL(Dataset):
    def __init__(self, name):
        self.name = name
        self.train = TSP(data_dir='./data/TSP', split='train', num_neighbors=25, max_samples=10000) 
        self.val = TSP(data_dir='./data/TSP', split='val', num_neighbors=25, max_samples=1000)
        self.test = TSP(data_dir='./data/TSP', split='test', num_neighbors=25, max_samples=1000)
        

class TSPDataset(Dataset):
    def __init__(self, name, folder_prefix=''):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = folder_prefix + 'data/TSP/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.test = f[1]
            self.val = f[2]
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))
    
    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        # Edge classification labels need to be flattened to 1D lists
        labels = torch.LongTensor(np.array(list(itertools.chain(*labels))))
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = torch.cat(tab_snorm_n).sqrt()  
        #tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        #tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        #snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels
    
    
    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense_gnn(self, samples, edge_feat):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        # Edge classification labels need to be flattened to 1D lists
        labels = torch.LongTensor(np.array(list(itertools.chain(*labels))))
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = tab_snorm_n[0][0].sqrt()  
        
        #batched_graph = dgl.batch(graphs)
        
        g = graphs[0]
        adj = self._sym_normalize_adj(g.adjacency_matrix().to_dense())        
        """
            Adapted from https://github.com/leichen2018/Ring-GNN/
            Assigning node and edge feats::
            we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
            Then we build a zero-initialized tensor, say T, in R^{(1 + d_n + d_e) x n x n}. T[0, :, :] is the adjacency matrix.
            The diagonal T[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i. 
            The off diagonal T[1+d_n:, i, j] store edge features of edge(i, j).
        """

        zero_adj = torch.zeros_like(adj)

        in_node_dim = g.ndata['feat'].shape[1]
        in_edge_dim = g.edata['feat'].shape[1]
        
        if edge_feat:
            # use edge feats also to prepare adj
            adj_with_edge_feat = torch.stack([zero_adj for j in range(in_node_dim + in_edge_dim)])
            adj_with_edge_feat = torch.cat([adj.unsqueeze(0), adj_with_edge_feat], dim=0)

            us, vs = g.edges()      
            for idx, edge_feat in enumerate(g.edata['feat']):
                adj_with_edge_feat[1+in_node_dim:, us[idx], vs[idx]] = edge_feat

            for node, node_feat in enumerate(g.ndata['feat']):
                adj_with_edge_feat[1:1+in_node_dim, node, node] = node_feat
            
            x_with_edge_feat = adj_with_edge_feat.unsqueeze(0)
            
            return None, x_with_edge_feat, labels, g.edges()
        else:
            # use only node feats to prepare adj
            adj_no_edge_feat = torch.stack([zero_adj for j in range(in_node_dim)])
            adj_no_edge_feat = torch.cat([adj.unsqueeze(0), adj_no_edge_feat], dim=0)

            for node, node_feat in enumerate(g.ndata['feat']):
                adj_no_edge_feat[1:1+in_node_dim, node, node] = node_feat

            x_no_edge_feat = adj_no_edge_feat.unsqueeze(0)
        
            return x_no_edge_feat, None, labels, g.edges()
    
    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim = 0)#.squeeze()
        deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))
    
    
    def _add_self_loops(self):
        """
           No self-loop support since TSP edge classification dataset. 
        """
        raise NotImplementedError
        
        