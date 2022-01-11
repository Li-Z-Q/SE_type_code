import torch
import torch.nn as nn
from torch import optim


class MyModel(nn.Module):
    def __init__(self, dropout):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.l = nn.Linear(in_features=2, out_features=3)

        self.test = nn.Parameter(torch.tensor([1.2]))

    def forward(self):
        loss = self.test * 4
        return loss


model = MyModel(dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

for name, parameters in model.named_parameters():
    print(name, ':', parameters)
    # print()
    break

# loss = model.forward()
# loss.backward()
# optimizer.step()
# print("after update --------------------------------------------")
# for name, parameters in model.named_parameters():
#     print(name, ':', parameters)
#     # print()
#     break
# torch.save(model, './test_model.pt')
# print("save")

model_new = torch.load('./test_model.pt')
for name, parameters in model_new.named_parameters():
    print(name, ':', parameters)
    # print()
    break