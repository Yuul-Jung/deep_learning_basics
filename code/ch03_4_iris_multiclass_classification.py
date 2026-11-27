# ch03_4_iris_multiclass_classification.py — 원본: lab_Pytorch/ch03_4_iris_multiclass_classification.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

data_path = Path("iris.csv")

if not data_path.exists():
    try:
        from google.colab import files
        files.upload()
    except ImportError:
        pass

if not data_path.exists():
    raise FileNotFoundError(
        "현재 폴더에 iris.csv를 넣은 뒤 다시 실행해 주세요."
    )

iris_data = pd.read_csv(data_path)

print("데이터 shape:", iris_data.shape)
print("열 이름:", iris_data.columns.tolist())

# %% CODE 02
print("앞부분 5행:")
print(iris_data.head())

print()
print("결측값 수:", iris_data.isna().sum().sum())
print("중복 행 수:", iris_data.duplicated().sum())

print()
print("품종별 데이터 수:")
print(iris_data["label"].value_counts())

# %% CODE 03
class_names = [
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica",
]

plt.figure(figsize=(8, 5))

for class_name in class_names:
    class_rows = iris_data[
        iris_data["label"] == class_name
    ]

    plt.scatter(
        class_rows["F3"],
        class_rows["F4"],
        label=class_name,
    )

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Data")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# %% CODE 04
feature_columns = ["F1", "F2", "F3", "F4"]

label_to_index = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}

index_to_label = {
    index: label
    for label, index in label_to_index.items()
}

X = iris_data[
    feature_columns
].astype(np.float32)

y = iris_data["label"].map(
    label_to_index
).astype(np.int64)

print("입력 shape:", X.shape)
print("정답 shape:", y.shape)
print("정답 클래스:", sorted(y.unique()))

# %% CODE 05
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("훈련 데이터:", X_train.shape, y_train.shape)
print("테스트 데이터:", X_test.shape, y_test.shape)

print()
print("훈련 클래스별 개수:")
print(y_train.value_counts().sort_index())

print()
print("테스트 클래스별 개수:")
print(y_test.value_counts().sort_index())

# %% CODE 06
train_mean = X_train.mean()
train_std = X_train.std()

X_train_scaled = (
    X_train - train_mean
) / train_std

X_test_scaled = (
    X_test - train_mean
) / train_std

print("표준화 후 훈련 평균:")
print(X_train_scaled.mean().round(3))

print()
print("표준화 후 훈련 표준편차:")
print(X_train_scaled.std().round(3))

# %% CODE 07
X_train_tensor = torch.tensor(
    X_train_scaled.to_numpy(),
    dtype=torch.float32,
)

X_test_tensor = torch.tensor(
    X_test_scaled.to_numpy(),
    dtype=torch.float32,
)

y_train_tensor = torch.tensor(
    y_train.to_numpy(),
    dtype=torch.long,
)

y_test_tensor = torch.tensor(
    y_test.to_numpy(),
    dtype=torch.long,
)

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

model = nn.Linear(4, 3)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
)

print(model)
print("가중치 shape:", model.weight.shape)
print("편향 shape:", model.bias.shape)

# %% CODE 09
epochs = 1500
loss_history = []

model.train()

for epoch in range(epochs):
    # 1. 클래스별 로짓 계산
    logits = model(
        X_train_tensor
    )

    # 2. 교차엔트로피 손실 계산
    loss = loss_function(
        logits,
        y_train_tensor,
    )

    # 3. 이전 기울기 초기화
    optimizer.zero_grad()

    # 4. 역전파
    loss.backward()

    # 5. 가중치와 편향 갱신
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
plt.ylabel("Cross Entropy Loss")
plt.title("Training Loss")
plt.grid(alpha=0.3)
plt.show()

# %% CODE 11
model.eval()

with torch.no_grad():
    train_logits = model(
        X_train_tensor
    )

    test_logits = model(
        X_test_tensor
    )

    train_predictions = torch.argmax(
        train_logits,
        dim=1,
    )

    test_predictions = torch.argmax(
        test_logits,
        dim=1,
    )

    test_probabilities = torch.softmax(
        test_logits,
        dim=1,
    )

    train_accuracy = (
        train_predictions == y_train_tensor
    ).float().mean()

    test_accuracy = (
        test_predictions == y_test_tensor
    ).float().mean()

print(f"훈련 정확도: {train_accuracy.item():.4f}")
print(f"테스트 정확도: {test_accuracy.item():.4f}")

# %% CODE 12
print(
    "번호  | 실제 품종            | "
    "예측 품종            | 확률 [setosa, versicolor, virginica]"
)

for index in range(10):
    actual = y_test_tensor[index].item()
    predicted = test_predictions[index].item()

    probabilities = [
        round(value, 4)
        for value in test_probabilities[index].tolist()
    ]

    print(
        f"{index:>4d} | "
        f"{index_to_label[actual]:<18} | "
        f"{index_to_label[predicted]:<18} | "
        f"{probabilities}"
    )

# %% CODE 13
number_of_classes = 3

confusion_matrix = torch.zeros(
    number_of_classes,
    number_of_classes,
    dtype=torch.int64,
)

for actual, predicted in zip(
    y_test_tensor,
    test_predictions,
):
    confusion_matrix[
        actual,
        predicted,
    ] += 1

print("테스트 혼동행렬:")
print(confusion_matrix)

short_names = [
    "setosa",
    "versicolor",
    "virginica",
]

plt.figure(figsize=(6, 5))
plt.imshow(confusion_matrix.numpy())

plt.xticks(
    range(number_of_classes),
    short_names,
    rotation=20,
)

plt.yticks(
    range(number_of_classes),
    short_names,
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Test Confusion Matrix")

for row in range(number_of_classes):
    for column in range(number_of_classes):
        plt.text(
            column,
            row,
            int(confusion_matrix[row, column]),
            ha="center",
            va="center",
            fontsize=13,
        )

plt.colorbar()
plt.tight_layout()
plt.show()

# %% CODE 14
for class_index, class_name in enumerate(
    class_names
):
    class_mask = (
        y_test_tensor == class_index
    )

    class_accuracy = (
        test_predictions[class_mask]
        == y_test_tensor[class_mask]
    ).float().mean()

    print(
        f"{class_name:<18}: "
        f"{class_accuracy.item():.4f}"
    )

# %% CODE 15
wrong_indices = torch.where(
    test_predictions
    != y_test_tensor
)[0]

if len(wrong_indices) == 0:
    print("잘못 분류한 테스트 데이터가 없습니다.")
else:
    print("잘못 분류한 테스트 데이터:")

    for index in wrong_indices:
        index = index.item()

        actual = y_test_tensor[index].item()
        predicted = test_predictions[index].item()

        print(
            f"입력 {X_test.iloc[index].tolist()} | "
            f"실제 {index_to_label[actual]} | "
            f"예측 {index_to_label[predicted]}"
        )

# %% CODE 16
new_flowers = pd.DataFrame(
    [
        [5.1, 3.5, 1.4, 0.2],
        [6.0, 2.9, 4.5, 1.5],
        [6.7, 3.1, 5.6, 2.4],
    ],
    columns=feature_columns,
    dtype=np.float32,
)

new_flowers_scaled = (
    new_flowers - train_mean
) / train_std

new_flowers_tensor = torch.tensor(
    new_flowers_scaled.to_numpy(),
    dtype=torch.float32,
)

model.eval()

with torch.no_grad():
    new_logits = model(
        new_flowers_tensor
    )

    new_probabilities = torch.softmax(
        new_logits,
        dim=1,
    )

    new_predictions = torch.argmax(
        new_logits,
        dim=1,
    )

for flower, probabilities, predicted in zip(
    new_flowers.to_numpy(),
    new_probabilities,
    new_predictions,
):
    probability_list = [
        round(value, 4)
        for value in probabilities.tolist()
    ]

    print(
        f"입력 {flower.tolist()} "
        f"→ {index_to_label[predicted.item()]} "
        f"| 확률 {probability_list} "
    )
