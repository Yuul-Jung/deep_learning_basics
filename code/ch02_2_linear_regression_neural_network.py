# ch02_2_linear_regression_neural_network.py — 원본: lab_Pytorch/ch02_2_linear_regression_neural_network.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
import torch
from torch import nn
torch.manual_seed(42)
x_train = torch.tensor([[1.0], [2.0], [3.0]])
y_train = torch.tensor([[1.0], [2.0], [3.0]])
print("x_train:")
print(x_train)
print("shape:", x_train.shape)
print()
print("y_train:")
print(y_train)
print("shape:", y_train.shape)

# %% CODE 02
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 입력 1개 → 출력 1개
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        # 내부에서 y_hat = W*x + b 계산
        return self.linear(x)
model = LinearRegressionModel()
print(model)

# %% CODE 03
print("초기 W:", model.linear.weight.item())
print("초기 b:", model.linear.bias.item())
print()
for name, parameter in model.named_parameters():
    print(name, parameter.shape)

# %% CODE 04
def train_model(model, x, y, learning_rate=0.01, epochs=2000):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )
    first_loss = None
    for epoch in range(epochs):
        # 1. 순전파
        prediction = model(x)
        # 2. 손실 계산
        loss = loss_function(prediction, y)
        if epoch == 0:
            first_loss = loss.item()
        # 3. 이전 기울기 초기화
        optimizer.zero_grad()
        # 4. 역전파
        loss.backward()
        # 5. 매개변수 갱신
        optimizer.step()
    return first_loss, loss.item()

# %% CODE 05
torch.manual_seed(42)
model = LinearRegressionModel()
first_loss, final_loss = train_model(
    model,
    x_train,
    y_train,
)
print("학습된 W:", model.linear.weight.item())
print("학습된 b:", model.linear.bias.item())
print("Loss 변화:", first_loss, "→", final_loss)

# %% CODE 06
x_new = torch.tensor([[2.5], [5.0]])

model.eval()

with torch.no_grad():
    y_new = model(x_new)

print("새 입력:")
print(x_new[:, 0])

print()
print("예측값:")
print(y_new[:, 0])

# %% CODE 07
voltage_data = torch.tensor(
    [[0.5], [1.0], [1.5], [2.0]]
)

measured_data = torch.tensor(
    [[11.0], [21.0], [31.0], [41.0]]
)

torch.manual_seed(42)

sensor_model = LinearRegressionModel()

sensor_first_loss, sensor_final_loss = train_model(
    sensor_model,
    voltage_data,
    measured_data,
    learning_rate=0.05,
    epochs=3000,
)

new_voltage = torch.tensor([[1.25]])

sensor_model.eval()

with torch.no_grad():
    corrected_value = sensor_model(new_voltage)

print("센서 W:", sensor_model.linear.weight.item())
print("센서 b:", sensor_model.linear.bias.item())
print("Loss 변화:", sensor_first_loss, "→", sensor_final_loss)
print("1.25 V 보정값:", corrected_value.item())
