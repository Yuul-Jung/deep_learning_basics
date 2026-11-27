# ch02_5_multi_variable_linear_regression.py — 원본: lab_Pytorch/ch02_5_multi_variable_linear_regression.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
import torch
from torch import nn
import matplotlib.pyplot as plt
x_train = torch.tensor(
    [
        [73.0, 80.0, 75.0],
        [93.0, 88.0, 93.0],
        [89.0, 91.0, 90.0],
        [96.0, 98.0, 100.0],
        [73.0, 66.0, 70.0],
    ],
    dtype=torch.float32,
)
y_train = torch.tensor(
    [[152.0], [185.0], [180.0], [196.0], [142.0]],
    dtype=torch.float32,
)
print("입력 shape:", x_train.shape)
print("목표값 shape:", y_train.shape)

# %% CODE 02
# 첫 번째 학생을 예로 직접 계산합니다.
first_student = x_train[0]
sample_weight = torch.tensor([0.5, 0.8, 0.7])
sample_bias = 10.0
sample_prediction = (
    first_student * sample_weight
).sum() + sample_bias
print("첫 번째 학생 점수:", first_student)
print("예시 예측값:", sample_prediction.item())

# %% CODE 03
# 각 열의 평균과 표준편차
x_mean = x_train.mean(dim=0)
x_std = x_train.std(dim=0)
# 각 열을 따로 표준화
x_train_scaled = (
    x_train - x_mean
) / x_std
print("특성별 평균:", x_train_scaled.mean(dim=0))
print("특성별 표준편차:", x_train_scaled.std(dim=0))

# %% CODE 04
torch.manual_seed(42)
model = nn.Linear(3, 1)
print(model)
print("가중치 shape:", model.weight.shape)
print("편향 shape:", model.bias.shape)

# %% CODE 05
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
    loss = loss_function(
        prediction,
        y_train,
    )
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
model.eval()
with torch.no_grad():
    predicted_score = model(
        x_train_scaled
    )
for actual, predicted in zip(
    y_train,
    predicted_score,
):
    print(
        f"실제: {actual.item():6.1f} | "
        f"예측: {predicted.item():6.1f}"
    )

# %% CODE 08
scaled_weight = model.weight.detach().squeeze(0)
scaled_bias = model.bias.detach().squeeze(0)
# 표준화 이전 단위의 가중치와 편향
real_weight = scaled_weight / x_std
real_bias = (
    scaled_bias
    - torch.sum(
        real_weight * x_mean
    )
)
print(
    "예측식:"
)
print(
    f"y = {real_weight[0].item():.3f}x1 "
    f"{real_weight[1].item():+.3f}x2 "
    f"{real_weight[2].item():+.3f}x3 "
    f"{real_bias.item():+.3f}"
)

# %% CODE 09
error = predicted_score - y_train
mae = error.abs().mean()
rmse = (error ** 2).mean().sqrt()
residual_sum = (error ** 2).sum()
total_sum = (
    (y_train - y_train.mean()) ** 2
).sum()
r2 = 1 - residual_sum / total_sum
print(f"MAE : {mae.item():.3f}점")
print(f"RMSE: {rmse.item():.3f}점")
print(f"R² : {r2.item():.6f}")

# %% CODE 10
new_scores = torch.tensor(
    [
        [85.0, 90.0, 88.0],
        [95.0, 90.0, 92.0],
        [70.0, 75.0, 80.0],
    ],
    dtype=torch.float32,
)
new_scores_scaled = (
    new_scores - x_mean
) / x_std
with torch.no_grad():
    new_prediction = model(
        new_scores_scaled
    )
for scores, predicted in zip(
    new_scores,
    new_prediction,
):
    print(
        f"입력 {scores.tolist()} "
        f"→ 예상 최종 점수 "
        f"{predicted.item():.1f}"
    )

# %% CODE 11
practice_scores = torch.tensor(
    [
        [80.0, 80.0, 80.0],
        [90.0, 80.0, 80.0],
        [80.0, 90.0, 80.0],
        [80.0, 80.0, 90.0],
    ],
    dtype=torch.float32,
)
practice_scaled = (
    practice_scores - x_mean
) / x_std
with torch.no_grad():
    practice_prediction = model(
        practice_scaled
    )
base_score = practice_prediction[0].item()
for scores, predicted in zip(
    practice_scores,
    practice_prediction,
):
    change = predicted.item() - base_score
    print(
        f"{scores.tolist()} "
        f"→ {predicted.item():.1f}점 "
        f"(기준 대비 {change:+.1f})"
    )

# %% CODE 12
actual_values = y_train.squeeze().numpy()
predicted_values = (
    predicted_score.squeeze().numpy()
)
score_min = min(
    actual_values.min(),
    predicted_values.min(),
) - 3
score_max = max(
    actual_values.max(),
    predicted_values.max(),
) + 3
plt.figure(figsize=(7, 6))
plt.scatter(
    actual_values,
    predicted_values,
)
plt.plot(
    [score_min, score_max],
    [score_min, score_max],
    linestyle="--",
    label="Ideal Prediction",
)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
