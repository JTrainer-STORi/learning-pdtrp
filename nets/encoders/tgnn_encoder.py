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
        self.memory = nn.Parameter(torch.zeros(batch_size, num_nodes, memory_dim), requires_grad=False)
        self.gru = nn.GRUCell(message_dim, memory_dim)

    def get_memory(self, batch_id, node_ids):
        return self.memory[batch_id][node_ids]

    def update_memory(self, batch_id, node_ids, messages):
        old_mem = self.memory[batch_id][node_ids]
        new_mem = self.gru(messages, old_mem)
        self.memory.data[batch_id][node_ids] = new_mem

class MessageFunction(nn.Module):
  """
  Module which computes the message for a given interaction.
  """

  def compute_message(self, raw_messages):
    return None

class IdentityMessageFunction(MessageFunction):

  def compute_message(self, raw_messages):

    return raw_messages

class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
  """
  def __init__(self):
    super(MessageAggregator, self).__init__()

  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids, and a list of messages of the same length, aggregate different
    messages for the same id using one of the possible strategies.
    :param node_ids: A list of node ids of length batch_size
    :param messages: A tensor of shape [batch_size, message_length]
    :param timestamps A tensor of shape [batch_size]
    :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
    """

  def group_by_id(self, node_ids, messages, timestamps):
    node_id_to_messages = defaultdict(list)

    for i, node_id in enumerate(node_ids):
      node_id_to_messages[node_id].append((messages[i], timestamps[i]))

    return node_id_to_messages


class LastMessageAggregator(MessageAggregator):
  def __init__(self):
    super(LastMessageAggregator, self).__init__()

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""    
    unique_node_ids = np.unique(node_ids)
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
    super(MeanMessageAggregator, self).__init__()

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.mean(torch.stack(messages[node_id]), dim=0))
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


def get_message_aggregator(aggregator_type):
  if aggregator_type == "last":
    return LastMessageAggregator()
  elif aggregator_type == "mean":
    return MeanMessageAggregator()
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))