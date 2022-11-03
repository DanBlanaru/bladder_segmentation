import torch
from torch import nn
from monai.networks.nets import DenseNet121,UNet
from monai.networks.layers import Norm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_class = 2
batch = torch.randn(4,1,16,16,16).to(device)
batch2 = torch.randn(4,1,16,16,16).to(device)
# model = DenseNet121(spatial_dims=2, in_channels=1,
#                     out_channels=num_class).to(device)
model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
output = model.forward(batch)
target= torch.ones_like(output).to(device)
loss = loss_function(output,target)
optimizer.step()
out2 = model.forward(batch2)
loss2 = loss_function(out2,target)
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