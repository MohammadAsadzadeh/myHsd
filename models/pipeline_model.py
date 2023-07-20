from models import rulebased_model

class PiplineModel():
  def __init__(self, preproccessor, rulebased_model , bert_model , bert_tokenizer, device):
    self.preproccessor = preproccessor
    self.rulebased_model = rulebased_model
    self.bert_model = bert_model
    self.bert_tokenizer = bert_tokenizer
    self.device = device
  def __call__(self, input_text):
    t = self.preproccessor(input_text, "rule_based")
    r = self.rulebased_model(t)[0]
    if r == 1:
      return 1
    else:
      X = self.bert_tokenizer(self.preproccessor(input_text, "bert_based"), return_tensors="pt")
      b = self.bert_model(X.to(self.device))
      return b.item()