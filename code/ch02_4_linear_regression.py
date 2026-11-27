# ch02_4_linear_regression.py — 원본: lab_Pytorch/ch02_4_linear_regression.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
import torch
from torch import nn
import matplotlib.pyplot as plt

# 참돔 길이(cm)
fish_length = torch.tensor(
    [
        8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0,
        19.6, 20.0, 21.0, 21.0, 21.0, 21.3, 22.0, 22.0,
        22.0, 22.0, 22.0, 22.5, 22.5, 22.7, 23.0, 23.5,
        24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5,
        27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
        36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0,
        40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5, 44.0,
    ],
    dtype=torch.float32,
).reshape(-1, 1)

# 참돔 무게(g)
fish_weight = torch.tensor(
    [
        5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0,
        85.0, 85.0, 110.0, 115.0, 125.0, 130.0, 120.0, 120.0,
        130.0, 135.0, 110.0, 130.0, 150.0, 145.0, 150.0, 170.0,
        225.0, 145.0, 188.0, 180.0, 197.0, 218.0, 300.0, 260.0,
        265.0, 250.0, 250.0, 300.0, 320.0, 514.0, 556.0, 840.0,
        685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0, 850.0,
        900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0, 1000.0,
    ],
    dtype=torch.float32,
).reshape(-1, 1)

print("길이 shape:", fish_length.shape)
print("무게 shape:", fish_weight.shape)
print("길이 범위:", fish_length.min().item(), "~", fish_length.max().item(), "cm")

# %% CODE 02
plt.figure(figsize=(8, 5))
plt.scatter(fish_length.numpy(), fish_weight.numpy())
plt.xlabel("Length (cm)")
plt.ylabel("Weight (g)")
plt.title("Fish Length and Weight")
plt.grid(alpha=0.3)
plt.show()

# %% CODE 03
torch.manual_seed(42)

indices = torch.randperm(len(fish_length))

test_indices = indices[:14]
train_indices = indices[14:]

x_train = fish_length[train_indices]
y_train = fish_weight[train_indices]

x_test = fish_length[test_indices]
y_test = fish_weight[test_indices]

print("훈련 데이터:", x_train.shape, y_train.shape)
print("테스트 데이터:", x_test.shape, y_test.shape)

# %% CODE 04
x_mean = x_train.mean()
x_std = x_train.std()

x_train_scaled = (x_train - x_mean) / x_std
x_test_scaled = (x_test - x_mean) / x_std

print("표준화 전 평균:", x_train.mean().item())
print("표준화 후 평균:", x_train_scaled.mean().item())
print("표준화 후 표준편차:", x_train_scaled.std().item())

# %% CODE 05
torch.manual_seed(42)
model = nn.Linear(1, 1)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05,
)
epochs = 1500
loss_history = []
for epoch in range(epochs):
    # 1. 순전파
    prediction = model(x_train_scaled)
    # 2. 손실 계산
    loss = loss_function(prediction, y_train)
    # 3. 이전 기울기 초기화
    optimizer.zero_grad()
    # 4. 역전파
    loss.backward()
    # 5. 가중치와 편향 갱신
    optimizer.step()
    loss_history.append(loss.item())
print("처음 Loss:", loss_history[0])
print("마지막 Loss:", loss_history[-1])

# %% CODE 06
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid(alpha=0.3)
plt.show()

# %% CODE 07
scaled_weight = model.weight.item()
scaled_bias = model.bias.item()
real_weight = scaled_weight / x_std.item()
real_bias = (
    scaled_bias
    - scaled_weight * x_mean.item() / x_std.item()
)
print(
    f"학습된 회귀식: "
    f"무게 = {real_weight:.2f} × 길이 {real_bias:+.2f}"
)
print(f"기울기: {real_weight:.2f} g/cm")
print(f"절편: {real_bias:.2f} g")

# %% CODE 08
def regression_metrics(y_true, y_pred):
    error = y_pred - y_true

    mae = error.abs().mean()
    rmse = (error ** 2).mean().sqrt()

    residual_sum = (error ** 2).sum()
    total_sum = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - residual_sum / total_sum

    return mae, rmse, r2


model.eval()

with torch.no_grad():
    train_prediction = model(x_train_scaled)
    test_prediction = model(x_test_scaled)

train_mae, train_rmse, train_r2 = regression_metrics(
    y_train, train_prediction,)

test_mae, test_rmse, test_r2 = regression_metrics(
    y_test,test_prediction,)

print("[훈련 데이터]")
print(f"MAE : {train_mae.item():.2f} g")
print(f"RMSE: {train_rmse.item():.2f} g")
print(f"R²  : {train_r2.item():.4f}")

print()
print("[테스트 데이터]")
print(f"MAE : {test_mae.item():.2f} g")
print(f"RMSE: {test_rmse.item():.2f} g")
print(f"R²  : {test_r2.item():.4f}")

# %% CODE 09
new_lengths = torch.tensor([[25.0], [35.0], [40.0], [50.0]])

new_lengths_scaled = (new_lengths - x_mean) / x_std

with torch.no_grad():
    new_weights = model(new_lengths_scaled)

for length, weight in zip(new_lengths, new_weights):
    print(f"{length.item():.0f} cm "
          f"→ {weight.item():.1f} g")

# %% CODE 10
line_x = torch.linspace(fish_length.min().item(),
    50.0, steps=100,).reshape(-1, 1)

line_x_scaled = (line_x - x_mean) / x_std

with torch.no_grad():
    line_y = model(line_x_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(fish_length.numpy(),fish_weight.numpy(),label="Data",)
plt.plot(line_x.numpy(),line_y.numpy(),label="Linear Regression",)
plt.scatter([50.0],[new_weights[-1].item()], marker="^", 
            s=90, label="50 cm Prediction",)
plt.xlabel("Length (cm)")
plt.ylabel("Weight (g)")
plt.title("Linear Regression")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
