import torch
from torch import nn
from monai.networks.nets import DenseNet121
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_class = 2
batch = torch.randn(4,1,64,64).to(device)
model = DenseNet121(spatial_dims=2, in_channels=1,
                    out_channels=num_class).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
output = model(batch)
target= torch.ones_like(output).to(device)
loss = loss_function(output,target)
optimizer.step()


# device = torch.device("cuda")
# m = nn.Conv3d(16, 33, 3, stride=2)
# m= m.to(device)
# input = torch.randn(20, 16, 10, 50, 100)
# input = input.to(device)
# output = m(input)
# # dummy label
# target = torch.zeros_like(output)

# criterion = nn.MSELoss()
# opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
# loss = criterion(output, target)
# loss.backward()
# opt.step()
print(output.device)