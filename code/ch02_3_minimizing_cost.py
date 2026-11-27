# ch02_3_minimizing_cost.py — 원본: lab_Pytorch/ch02_3_minimizing_cost.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
import torch
import matplotlib.pyplot as plt

x_train = torch.tensor([1.0, 2.0, 3.0])
y_train = torch.tensor([1.0, 2.0, 3.0])

print("입력:", x_train)
print("실제값:", y_train)

# %% CODE 02
def calculate_cost(weight, x=x_train, y=y_train):
    prediction = weight * x
    return ((prediction - y) ** 2).mean().item()


for weight in [0.0, 1.0, 2.0]:
    cost = calculate_cost(weight)
    print(f"W={weight:.1f} → Cost={cost:.4f}")

# %% CODE 03
weight_values = torch.linspace(-3.0, 5.0, 161)
cost_values = [
    calculate_cost(weight.item())
    for weight in weight_values
]

plt.figure(figsize=(8, 5))
plt.plot(weight_values.numpy(), cost_values, label="Cost(W)")
plt.scatter([1.0], [0.0], label="Minimum W=1")
plt.xlabel("Weight W")
plt.ylabel("Cost")
plt.title("Cost Function")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# %% CODE 04
# W=5에서 기울기를 계산합니다.
weight = torch.tensor(5.0)

manual_gradient = 2 * (
    (weight * x_train - y_train) * x_train
).mean()

print("현재 W:", weight.item())
print("기울기:", manual_gradient.item())

# %% CODE 05
weight_auto = torch.tensor(
    5.0,
    requires_grad=True,
)

prediction = weight_auto * x_train
cost = ((prediction - y_train) ** 2).mean()

cost.backward()

print("직접 계산:", manual_gradient.item())
print("자동미분:", weight_auto.grad.item())

# %% CODE 06
learning_rate = 0.1

new_weight = weight - learning_rate * manual_gradient

print("이동 전 W:", weight.item())
print("이동 후 W:", new_weight.item())
print("이동 후 Cost:", calculate_cost(new_weight.item()))

# %% CODE 07
def minimize_cost(
    x,
    y,
    start_weight,
    learning_rate=0.1,
    epochs=20,
):
    # 학습할 가중치
    W = torch.tensor(
        float(start_weight),
        requires_grad=True,
    )

    optimizer = torch.optim.SGD(
        [W],
        lr=learning_rate,
    )

    weight_history = [W.item()]
    cost_history = [
        ((W * x - y) ** 2).mean().item()
    ]

    for _ in range(epochs):
        # 1. 순전파
        prediction = W * x

        # 2. 비용 계산
        cost = ((prediction - y) ** 2).mean()

        # 3. 기울기 초기화
        optimizer.zero_grad()

        # 4. 역전파
        cost.backward()

        # 5. 가중치 갱신
        optimizer.step()

        # 갱신된 결과 기록
        with torch.no_grad():
            new_cost = ((W * x - y) ** 2).mean()

        weight_history.append(W.item())
        cost_history.append(new_cost.item())

    return W.detach(), weight_history, cost_history

# %% CODE 08
final_weight, weight_history, cost_history = minimize_cost(
    x_train,
    y_train,
    start_weight=5.0,
    learning_rate=0.1,
    epochs=20,
)

print("시작 W:", weight_history[0])
print("최종 W:", final_weight.item())
print("시작 Cost:", cost_history[0])
print("최종 Cost:", cost_history[-1])

# %% CODE 09
plt.figure(figsize=(8, 5))
plt.plot(weight_values.numpy(), cost_values, label="Cost(W)")
plt.scatter(
    weight_history,
    cost_history,
    label="Gradient Descent",
)
plt.xlabel("Weight W")
plt.ylabel("Cost")
plt.title("Movement of Weight")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# %% CODE 10
for rate in [0.01, 0.1, 0.3]:
    result_W, _, result_costs = minimize_cost(
        x_train,
        y_train,
        start_weight=5.0,
        learning_rate=rate,
        epochs=10,
    )

    print(
        f"학습률={rate:.2f} → "
        f"최종 W={result_W.item():.4f}, "
        f"최종 Cost={result_costs[-1]:.4f}"
    )

# %% CODE 11
sensor_x = torch.tensor([1.0, 2.0, 3.0])
sensor_y = torch.tensor([2.0, 4.0, 6.0])

sensor_W, _, sensor_costs = minimize_cost(
    sensor_x,
    sensor_y,
    start_weight=-3.0,
    learning_rate=0.1,
    epochs=20,
)

new_input = torch.tensor(4.0)

with torch.no_grad():
    new_output = sensor_W * new_input

print("학습된 W:", sensor_W.item())
print("최종 Cost:", sensor_costs[-1])
print("입력 4의 예측값:", new_output.item())
