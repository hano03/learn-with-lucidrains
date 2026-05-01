import torch
from torch import nn

# toy model

model = nn.Linear(10, 1)

# import AdamAtan2 and instantiate with parameters

from adam_atan2_pytorch import AdamAtan2

opt = AdamAtan2(model.parameters(), lr = 1e-1) #经过测试，本 demo 下的1e-1学习率对loss收敛效果影响最佳

x = torch.randn(10)        # 固定的输入特征
target = torch.tensor([1.0]) # 假设我们希望模型的输出逼近 1.0

criterion = nn.MSELoss()

# forward and backwards

for i in range(100):
  pred = model(x)

  # loss = model(torch.randn(10))
  loss = criterion(pred, target)

  loss.backward()

  # optimizer step

  opt.step()
  opt.zero_grad()

  if i % 10 == 0:
    print(f"Step {i}: Loss = {loss.item():.4f}, Prediction = {pred.item():.4f}")