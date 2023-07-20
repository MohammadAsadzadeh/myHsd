
from transformers import AutoConfig, AutoTokenizer, AutoModel
from torch import nn
import torch

class BertBasedModel(nn.Module):
  def __init__(self, bert_path, bert_non_trainable=None, head_mask=None):
    super(BertBasedModel, self).__init__()
    #config = AutoConfig.from_pretrained(bert_path)
    self.head_mask = head_mask
    self.bert = AutoModel.from_pretrained(bert_path)
    for name, param in self.bert.named_parameters():
      param.requires_grad = True
      if "embedding" in name or "pooler" in name:
        param.requires_grad = False
      if bert_non_trainable:
        for i in bert_non_trainable:
          if 'encoder.layer.'+str(i)+"." in name:
            param.requires_grad = False
    self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.bert.config.hidden_size, self.bert.config.num_attention_heads, batch_first=True),1)
    self.linear1 = nn.Linear(self.bert.config.hidden_size, 1)

  def forward(self, x):
    x = self.bert(**x, head_mask=self.head_mask).last_hidden_state[:, 0, :]
    x = self.encoder(x)
    x = self.linear1(x)
    # x = self.dropout(x)
    # x = self.linear2(x)
    # x = self.head(x)
    return torch.sigmoid(x[:,0])