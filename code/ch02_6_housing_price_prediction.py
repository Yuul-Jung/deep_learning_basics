# ch02_6_housing_price_prediction.py — 원본: lab_Pytorch/ch02_6_housing_price_prediction.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
torch.manual_seed(42)
data_path = Path("housing.csv")
# Colab에서 파일이 없으면 업로드 창을 엽니다.
if not data_path.exists():
    try:
        from google.colab import files
        files.upload()
    except ImportError:
        pass
if not data_path.exists():
    raise FileNotFoundError(
        "현재 폴더에 housing.csv를 넣은 뒤 다시 실행해 주세요."
    )

# %% CODE 02
# 첫 줄에서 열 이름 읽기
with data_path.open("r", encoding="utf-8") as file:
    column_names = file.readline().strip().split(",")
# 숫자 데이터 읽기
housing_array = np.loadtxt(
    data_path,
    delimiter=",",
    skiprows=1,
    dtype=np.float32,
)
feature_names = column_names[:-1]
target_name = column_names[-1]
print("데이터 수:", len(housing_array))
print("입력 특성 수:", len(feature_names))
print("목표값:", target_name)
print("앞부분 3행:")
print(housing_array[:3])

# %% CODE 03
# NumPy 배열을 Tensor로 변환
housing_tensor = torch.tensor(
    housing_array,
    dtype=torch.float32,
)
x_all = housing_tensor[:, :-1]
y_all = housing_tensor[:, -1:]
print("전체 입력 shape:", x_all.shape)
print("전체 목표 shape:", y_all.shape)
print("첫 번째 입력:", x_all[0])
print("첫 번째 목표값:", y_all[0].item())

# %% CODE 04
torch.manual_seed(42)
indices = torch.randperm(len(x_all))
train_count = int(len(x_all) * 0.8)
train_indices = indices[:train_count]
test_indices = indices[train_count:]
x_train = x_all[train_indices]
y_train = y_all[train_indices]
x_test = x_all[test_indices]
y_test = y_all[test_indices]
print("훈련 데이터:", x_train.shape, y_train.shape)
print("테스트 데이터:", x_test.shape, y_test.shape)

# %% CODE 05
# 각 입력 특성의 평균과 표준편차
x_mean = x_train.mean(dim=0)
x_std = x_train.std(dim=0)

# 표준편차가 0인 특성이 있으면 1로 대체
x_std = torch.where(x_std == 0,torch.ones_like(x_std),x_std,)
x_train_scaled = (x_train - x_mean) / x_std
x_test_scaled = (x_test - x_mean) / x_std

print("표준화 후 평균:",torch.round(x_train_scaled.mean(dim=0),decimals=3,),)
print("표준화 후 표준편차:",torch.round(x_train_scaled.std(dim=0),decimals=3,),)

# %% CODE 06
torch.manual_seed(42)
model = nn.Linear(
    len(feature_names),
    1,
)
print(model)
print("가중치 shape:", model.weight.shape)
print("편향 shape:", model.bias.shape)

# %% CODE 07
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.03,
)
epochs = 2000
loss_history = []
model.train()
for epoch in range(epochs):
    # 1. 순전파
    train_prediction = model(
        x_train_scaled
    )
    # 2. 손실 계산
    loss = loss_function(
        train_prediction,
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

# %% CODE 08
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid(alpha=0.3)
plt.show()

# %% CODE 09
model.eval()
with torch.no_grad():
    train_prediction = model(
        x_train_scaled
    )
    test_prediction = model(
        x_test_scaled
    )
print("테스트 실제값 5개:")
print(y_test[:5, 0])
print()
print("테스트 예측값 5개:")
print(
    torch.round(
        test_prediction[:5, 0],
        decimals=2,
    )
)

# %% CODE 10
def regression_metrics(y_true, y_pred):
    error = y_pred - y_true
    mae = error.abs().mean()
    rmse = (error ** 2).mean().sqrt()
    residual_sum = (error ** 2).sum()
    total_sum = (
        (y_true - y_true.mean()) ** 2
    ).sum()
    r2 = 1 - residual_sum / total_sum
    return mae, rmse, r2
train_mae, train_rmse, train_r2 = (
    regression_metrics(
        y_train,
        train_prediction,
    )
)
test_mae, test_rmse, test_r2 = (
    regression_metrics(
        y_test,
        test_prediction,
    )
)
print("[훈련 데이터: 학습 적합도]")
print(f"MAE : {train_mae.item():.3f} 천 달러")
print(f"RMSE: {train_rmse.item():.3f} 천 달러")
print(f"R² : {train_r2.item():.3f}")
print()
print("[테스트 데이터: 일반화 성능]")
print(f"MAE : {test_mae.item():.3f} 천 달러")
print(f"RMSE: {test_rmse.item():.3f} 천 달러")
print(f"R² : {test_r2.item():.3f}")

# %% CODE 11
rmse_gap = (
    test_rmse - train_rmse
).item()
print(
    "RMSE 차이(테스트 - 훈련):",
    round(rmse_gap, 3),
)
if test_rmse > train_rmse * 1.3:
    print(
        "해석: 테스트 오차 증가가 커서 "
        "과적합 가능성을 점검합니다."
    )
else:
    print(
        "해석: 훈련과 테스트 오차 사이에 "
        "큰 급증은 없습니다."
    )

# %% CODE 12
actual_values = y_test[:, 0].numpy()
predicted_values = (
    test_prediction[:, 0].numpy()
)
price_min = min(
    actual_values.min(),
    predicted_values.min(),
)
price_max = max(
    actual_values.max(),
    predicted_values.max(),
)
plt.figure(figsize=(7, 6))
plt.scatter(
    actual_values,
    predicted_values,
    alpha=0.7,
)
plt.plot(
    [price_min, price_max],
    [price_min, price_max],
    linestyle="--",
    label="Ideal Prediction",
)
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# %% CODE 13
residuals = (
    actual_values - predicted_values
)
plt.figure(figsize=(8, 5))
plt.scatter(
    predicted_values,
    residuals,
    alpha=0.7,
)
plt.axhline(
    0,
    linestyle="--",
)
plt.xlabel("Predicted MEDV")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.grid(alpha=0.3)
plt.show()

# %% CODE 14
weights = (
    model.weight
    .detach()
    .squeeze(0)
)
weight_order = torch.argsort(
    weights.abs(),
    descending=True,
)
print("절댓값이 큰 가중치부터 확인:")
for index in weight_order:
    feature_index = index.item()
    print(
        f"{feature_names[feature_index]:8s}: "
        f"{weights[feature_index].item(): .4f}"
    )

# %% CODE 15
sample_index = 0
sample_actual = (
    y_test[sample_index].item()
)
sample_prediction = (
    test_prediction[sample_index].item()
)
sample_residual = (
    sample_actual - sample_prediction
)
print("입력 특성:")
for name, value in zip(
    feature_names,
    x_test[sample_index],
):
    print(
        f"{name:8s}: {value.item():.4f}"
    )
print()
print(
    f"실제 가격: {sample_actual:.2f} 천 달러"
)
print(
    f"예측 가격: {sample_prediction:.2f} 천 달러"
)
print(
    f"잔차    : {sample_residual:.2f} 천 달러"
)
