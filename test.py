from models.sentence_level_BiLSTM_bia_loss import MyModel


model = MyModel(dropout=0.5)

print(hasattr(model, 'dropout'))
print(hasattr(model, 'forward'))
print(hasattr(model, 'reset'))


import torch

a = torch.tensor([[1, 2, 3], [1, 4, 5]])
b = torch.tensor([[4, 5, 6], [9, 0, 98]])
print(a + b)

c = a
a = b
print(c)
