# ch02_1_linear_forward_and_loss.py — 원본: lab_Pytorch/ch02_1_linear_forward_and_loss.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
import torch
x_train = torch.tensor([1.0, 2.0, 3.0])
y_train = torch.tensor([1.0, 2.0, 3.0])
print("입력:", x_train)
print("실제값:", y_train)

# %% CODE 02
# 예시 직선: y = 0.5x + 2
W_example = 0.5
b_example = 2.0
y_pred = W_example * x_train + b_example
print("예측값:", y_pred)
print("실제값:", y_train)
print("오차:", y_pred - y_train)

# %% CODE 03
errors = y_pred - y_train
squared_errors = errors ** 2
mse = squared_errors.mean()
print("오차:", errors)
print("제곱 오차:", squared_errors)
print("MSE:", mse.item())

# %% CODE 04
# 첫 번째 직선: y = 1x + 0
pred_1 = 1.0 * x_train + 0.0
loss_1 = ((pred_1 - y_train) ** 2).mean()
# 두 번째 직선: y = 0.5x + 2
pred_2 = 0.5 * x_train + 2.0
loss_2 = ((pred_2 - y_train) ** 2).mean()
print("y = 1.0x + 0.0의 Loss:", loss_1.item())
print("y = 0.5x + 2.0의 Loss:", round(loss_2.item(), 4))

# %% CODE 05
def train_line(x, y, learning_rate=0.01, epochs=2000):
    # 학습할 가중치와 편향
    W = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.SGD([W, b], lr=learning_rate)
    first_loss = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        # 1. 순전파
        prediction = W * x + b
        # 2. 손실 계산
        loss = ((prediction - y) ** 2).mean()
        # 첫 번째 Loss 저장
        if epoch == 0:
            first_loss = loss.item()
        # 3. 역전파
        loss.backward()
        # 4. 매개변수 갱신
        optimizer.step()
    return W.detach(), b.detach(), first_loss, loss.item()

# %% CODE 06
W, b, first_loss, final_loss = train_line(x_train, y_train)
print("학습된 W:", W.item())
print("학습된 b:", b.item())
print("Loss 변화:", first_loss, "→", final_loss)

# %% CODE 07
x_new = torch.tensor([2.5, 5.0])
with torch.no_grad():
    y_new = W * x_new + b
print("새 입력:", x_new)
print("예측값:", y_new)

# %% CODE 08
voltage_data = torch.tensor([0.5, 1.0, 1.5, 2.0])
measured_data = torch.tensor([11.0, 21.0, 31.0, 41.0])
sensor_W, sensor_b, sensor_first_loss, sensor_final_loss = train_line(
    voltage_data,
    measured_data,
    learning_rate=0.05,
    epochs=3000,
)
new_voltage = torch.tensor(1.25)
with torch.no_grad():
    corrected_value = sensor_W * new_voltage + sensor_b
print("학습된 W:", sensor_W.item())
print("학습된 b:", sensor_b.item())
print("Loss 변화:", sensor_first_loss, "→", sensor_final_loss)
print("1.25 V 예측값:", corrected_value.item())
