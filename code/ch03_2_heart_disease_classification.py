# ch03_2_heart_disease_classification.py — 원본: lab_Pytorch/ch03_2_heart_disease_classification.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

# 두 파일 이름 중 현재 폴더에 있는 파일을 사용합니다.
data_path = Path("heart.csv")

if not data_path.exists():
    data_path = Path("heart(1).csv")

# Colab에서 파일이 없으면 업로드합니다.
if not data_path.exists():
    try:
        from google.colab import files
        files.upload()

        data_path = Path("heart.csv")

        if not data_path.exists():
            data_path = Path("heart(1).csv")

    except ImportError:
        pass

if not data_path.exists():
    raise FileNotFoundError(
        "현재 폴더에 heart.csv 또는 heart(1).csv를 넣어 주세요."
    )

heart_data = pd.read_csv(data_path)

print("사용 파일:", data_path.name)
print("데이터 shape:", heart_data.shape)

# %% CODE 02
print("앞부분 5행:")
print(heart_data.head())

print()
print("결측값 수:", heart_data.isna().sum().sum())
print("중복 행 수:", heart_data.duplicated().sum())

print()
print("target 클래스별 개수:")
print(
    heart_data["target"]
    .value_counts()
    .sort_index()
)

# %% CODE 03
X = heart_data.drop(
    columns="target"
).copy()

y = heart_data[["target"]].astype(
    np.float32
)

print("원본 입력 shape:", X.shape)
print("목표값 shape:", y.shape)

# %% CODE 04
continuous_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

categorical_columns = [
    "cp",
    "restecg",
    "slope",
    "ca",
    "thal",
]

X_encoded = pd.get_dummies(
    X,
    columns=categorical_columns,
    dtype=np.float32,
)

X_encoded = X_encoded.astype(
    np.float32
)

print("변환 전 shape:", X.shape)
print("변환 후 shape:", X_encoded.shape)
print("최종 입력 특성 수:", X_encoded.shape[1])

# %% CODE 05
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 이후 값 변경을 안전하게 하기 위해 복사합니다.
X_train = X_train.copy()
X_test = X_test.copy()

print("훈련 입력 shape:", X_train.shape)
print("테스트 입력 shape:", X_test.shape)

print()
print(
    "훈련 target=1 비율:",
    round(y_train["target"].mean(), 4),
)

print(
    "테스트 target=1 비율:",
    round(y_test["target"].mean(), 4),
)

# %% CODE 06
train_mean = X_train[
    continuous_columns
].mean()

train_std = X_train[
    continuous_columns
].std()

train_std = train_std.replace(
    0,
    1,
)

X_train.loc[:, continuous_columns] = (
    X_train[continuous_columns] - train_mean
) / train_std

X_test.loc[:, continuous_columns] = (
    X_test[continuous_columns] - train_mean
) / train_std

print("표준화 후 훈련 평균:")
print(
    X_train[continuous_columns]
    .mean()
    .round(3)
)

print()
print("표준화 후 훈련 표준편차:")
print(
    X_train[continuous_columns]
    .std()
    .round(3)
)

# %% CODE 07
X_train_tensor = torch.tensor(
    X_train.to_numpy(),
    dtype=torch.float32,
)

X_test_tensor = torch.tensor(
    X_test.to_numpy(),
    dtype=torch.float32,
)

y_train_tensor = torch.tensor(
    y_train.to_numpy(),
    dtype=torch.float32,
)

y_test_tensor = torch.tensor(
    y_test.to_numpy(),
    dtype=torch.float32,
)

input_size = X_train_tensor.shape[1]

print("입력 특성 수:", input_size)
print(
    "훈련 Tensor:",
    X_train_tensor.shape,
    y_train_tensor.shape,
)
print(
    "테스트 Tensor:",
    X_test_tensor.shape,
    y_test_tensor.shape,
)

# %% CODE 08
torch.manual_seed(42)

model = nn.Linear(
    input_size,
    1,
)

loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05,
)

print(model)
print("가중치 shape:", model.weight.shape)
print("편향 shape:", model.bias.shape)

# %% CODE 09
epochs = 2000
loss_history = []

model.train()

for epoch in range(epochs):
    # 1. 로짓 계산
    logits = model(
        X_train_tensor
    )

    # 2. BCE 손실 계산
    loss = loss_function(
        logits,
        y_train_tensor,
    )

    # 3. 이전 기울기 초기화
    optimizer.zero_grad()

    # 4. 역전파
    loss.backward()

    # 5. 가중치 갱신
    optimizer.step()

    loss_history.append(
        loss.item()
    )

print("처음 Loss:", loss_history[0])
print("마지막 Loss:", loss_history[-1])

# %% CODE 10
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title("Training Loss")
plt.grid(alpha=0.3)
plt.show()

# %% CODE 11
threshold = 0.5

model.eval()

with torch.no_grad():
    train_logits = model(
        X_train_tensor
    )

    test_logits = model(
        X_test_tensor
    )

    train_probabilities = torch.sigmoid(
        train_logits
    )

    test_probabilities = torch.sigmoid(
        test_logits
    )

    train_predictions = (
        train_probabilities >= threshold
    ).float()

    test_predictions = (
        test_probabilities >= threshold
    ).float()

print("테스트 예측 예시 10개")

for actual, probability, predicted in zip(
    y_test_tensor[:10],
    test_probabilities[:10],
    test_predictions[:10],
):
    print(
        f"실제 {int(actual.item())} | "
        f"확률 {probability.item():.4f} | "
        f"예측 {int(predicted.item())}"
    )

# %% CODE 12
def calculate_metrics(
    targets,
    predictions,
):
    targets = targets.reshape(-1)
    predictions = predictions.reshape(-1)

    tp = int(
        ((targets == 1) & (predictions == 1)).sum()
    )

    tn = int(
        ((targets == 0) & (predictions == 0)).sum()
    )

    fp = int(
        ((targets == 0) & (predictions == 1)).sum()
    )

    fn = int(
        ((targets == 1) & (predictions == 0)).sum()
    )

    accuracy = (
        tp + tn
    ) / len(targets)

    precision = tp / max(
        tp + fp,
        1,
    )

    recall = tp / max(
        tp + fn,
        1,
    )

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


train_metrics = calculate_metrics(
    y_train_tensor,
    train_predictions,
)

test_metrics = calculate_metrics(
    y_test_tensor,
    test_predictions,
)

print("[훈련 데이터]")

for name, value in train_metrics.items():
    print(f"{name:9s}: {value:.4f}")

print()
print("[테스트 데이터]")

for name, value in test_metrics.items():
    print(f"{name:9s}: {value:.4f}")

# %% CODE 13
confusion_matrix = np.array(
    [
        [
            test_metrics["TN"],
            test_metrics["FP"],
        ],
        [
            test_metrics["FN"],
            test_metrics["TP"],
        ],
    ]
)

plt.figure(figsize=(5, 4))

plt.imshow(
    confusion_matrix,
)

plt.xticks(
    [0, 1],
    ["Predicted 0", "Predicted 1"],
)

plt.yticks(
    [0, 1],
    ["Actual 0", "Actual 1"],
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Test Confusion Matrix")

for row in range(2):
    for column in range(2):
        plt.text(
            column,
            row,
            confusion_matrix[row, column],
            ha="center",
            va="center",
            fontsize=14,
        )

plt.colorbar()
plt.tight_layout()
plt.show()

# %% CODE 14
for threshold in [0.3, 0.5, 0.7]:
    threshold_predictions = (
        test_probabilities >= threshold
    ).float()

    metrics = calculate_metrics(
        y_test_tensor,
        threshold_predictions,
    )

    print(
        f"임계값 {threshold:.1f} | "
        f"Accuracy {metrics['Accuracy']:.4f} | "
        f"Precision {metrics['Precision']:.4f} | "
        f"Recall {metrics['Recall']:.4f} | "
        f"F1 {metrics['F1']:.4f}"
    )
