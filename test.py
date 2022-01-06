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

sim_matrix = [[10.1 for _ in range(7)] for __ in range(7)]
print(torch.tensor(sim_matrix).int())

sim_list = torch.tensor([1.2, 1.3])
s = torch.nn.Softmax(dim=0)
print(s(sim_list))
