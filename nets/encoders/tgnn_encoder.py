# New encoder experimenting with a temporal graph neural network (TGNN) encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class TimeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)

    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                       .float().reshape(dimension, -1))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    t = t.unsqueeze(dim=2)

    # output has shape [batch_size, seq_len, dimension]
    output = torch.cos(self.w(t))

    return output

class TGNEmbedding(nn.Module):
    def __init__(self, memory_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(memory_dim, embed_dim)
    def forward(self, mem):
        return F.relu(self.proj(mem))

class TGNMemory(nn.Module):
    def __init__(self, batch_size, num_nodes, memory_dim, message_dim):
        super().__init__()
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim

        # Not trainable, but persistent
        self.register_buffer("memory", torch.zeros(batch_size, num_nodes, memory_dim))
        self.gru = nn.GRUCell(message_dim, memory_dim)

    def get_memory(self, batch_id, node_ids):
        return self.memory[batch_id, node_ids].clone()

    def update_memory(self, batch_id, node_ids, messages):
        with torch.no_grad():
            old_memory = self.memory[batch_id, node_ids]
            new_memory = self.gru(messages, old_memory)
            self.memory[batch_id, node_ids] = new_memory


class MessageFunction(nn.Module):
  """
  Module which computes the message for a given interaction.
  """

  def compute_message(self, raw_messages):
    return None

class IdentityMessageFunction(MessageFunction):

  def compute_message(self, raw_messages):

    return raw_messages

class MessageAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def aggregate(self, node_ids, messages):
        raise NotImplementedError

class LastMessageAggregator(MessageAggregator):
  def __init__(self):
    super(LastMessageAggregator, self).__init__()

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""    
    unique_node_ids = torch.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    
    to_update_node_ids = []
    
    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            unique_messages.append(messages[node_id][-1][0])
            unique_timestamps.append(messages[node_id][-1][1])
    
    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps

class MeanMessageAggregator(MessageAggregator):
    def __init__(self):
        super().__init__()

    def aggregate(self, node_ids, messages):
        # node_ids: [B]
        # messages: [B, D]

        unique_ids = torch.unique(node_ids)
        aggregated_messages = []
        
        for uid in unique_ids:
            # Get mask for all messages targeting this node
            mask = (node_ids == uid)
            selected_msgs = messages[mask]  # [N_msgs_for_uid, D]
            mean_msg = selected_msgs.mean(dim=0)
            aggregated_messages.append(mean_msg)

        # Stack into a tensor
        aggregated_messages = torch.stack(aggregated_messages, dim=0)  # [num_unique_nodes, D]

        return unique_ids, aggregated_messages

def get_message_aggregator(aggregator_type):
  if aggregator_type == "last":
    return LastMessageAggregator()
  elif aggregator_type == "mean":
    return MeanMessageAggregator()
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))