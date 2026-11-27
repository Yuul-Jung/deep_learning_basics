# ch03_3_softmax_classification.py — 원본: lab_Pytorch/ch03_3_softmax_classification.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)

x_data = torch.tensor(
    [
        [1.0, 2.0, 1.0, 1.0],
        [2.0, 1.0, 3.0, 2.0],
        [3.0, 1.0, 3.0, 4.0],
        [4.0, 1.0, 5.0, 5.0],
        [1.0, 7.0, 5.0, 5.0],
        [1.0, 2.0, 5.0, 6.0],
        [1.0, 6.0, 6.0, 6.0],
        [1.0, 7.0, 7.0, 7.0],
    ],
    dtype=torch.float32,
)

# 0 → A, 1 → B, 2 → C
y_data = torch.tensor(
    [2, 2, 2, 1, 1, 1, 0, 0],
    dtype=torch.long,
)

class_names = ["A", "B", "C"]
number_of_classes = len(class_names)

print("입력 shape:", x_data.shape)
print("정답 shape:", y_data.shape)
print("정답 자료형:", y_data.dtype)

# %% CODE 02
y_one_hot = F.one_hot(
    y_data,
    num_classes=number_of_classes,
)

print("클래스 인덱스:")
print(y_data)

print()
print("원핫 벡터:")
print(y_one_hot)

# %% CODE 03
example_logits = torch.tensor(
    [
        [2.0, 1.0, 0.1],
        [-1.0, 0.5, 2.5],
    ],
    dtype=torch.float32,
)

# dim=1: 각 행의 A・B・C 로짓을 확률로 변환
example_probabilities = torch.softmax(
    example_logits,
    dim=1,
)

print("로짓:")
print(example_logits)

print()
print("Softmax 확률:")
print(example_probabilities)

print()
print("행별 확률 합:")
print(example_probabilities.sum(dim=1))

print()
print("예측 클래스 번호:")
print(torch.argmax(example_probabilities, dim=1))

# %% CODE 04
first_logit = example_logits[0:1]
first_target = torch.tensor([0])

first_probability = torch.softmax(
    first_logit,
    dim=1,
)

correct_probability = first_probability[0, 0]
manual_loss = -torch.log(correct_probability)

loss_function = nn.CrossEntropyLoss()
pytorch_loss = loss_function(
    first_logit,
    first_target,
)

print("정답 클래스 확률:", correct_probability.item())
print("직접 계산한 Loss:", manual_loss.item())
print("CrossEntropyLoss:", pytorch_loss.item())

# %% CODE 05
torch.manual_seed(42)

model = nn.Linear(4, 3)

print(model)
print("가중치 shape:", model.weight.shape)
print("편향 shape:", model.bias.shape)

initial_logits = model(x_data)
print("모델 출력 shape:", initial_logits.shape)

# %% CODE 06
loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
)

epochs = 2000
loss_history = []

model.train()

for epoch in range(epochs):
    # 1. 클래스별 로짓 계산
    logits = model(x_data)

    # 2. 교차엔트로피 손실 계산
    loss = loss_function(
        logits,
        y_data,
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

# %% CODE 07
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Training Loss")
plt.grid()
plt.show()

# %% CODE 08
model.eval()

with torch.no_grad():
    trained_logits = model(x_data)

    trained_probabilities = torch.softmax(
        trained_logits,
        dim=1,
    )

    predicted_classes = torch.argmax(
        trained_logits,
        dim=1,
    )

    accuracy = (
        predicted_classes == y_data
    ).float().mean()

print(f"학습 데이터 정확도: {accuracy.item():.4f}")

print()
print("번호 | 확률 [A, B, C]               | 예측 | 실제")

for index in range(len(x_data)):
    probabilities = trained_probabilities[index]
    predicted = predicted_classes[index].item()
    actual = y_data[index].item()

    probability_list = [
        round(value, 4)
        for value in probabilities.tolist()
    ]

    print(
        f"{index:>4d} | "
        f"{str(probability_list):<29} | "
        f"{class_names[predicted]:>4s} | "
        f"{class_names[actual]:>4s}"
    )

# %% CODE 09
plt.figure(figsize=(7, 5))
plt.imshow(
    trained_probabilities.numpy(),
    aspect="auto",
)

plt.xticks(
    range(number_of_classes),
    class_names,
)

plt.yticks(
    range(len(x_data)),
    range(len(x_data)),
)

plt.xlabel("Class")
plt.ylabel("Data Index")
plt.title("Class Probabilities")
plt.colorbar()

for row in range(len(x_data)):
    for column in range(number_of_classes):
        value = trained_probabilities[
            row,
            column,
        ].item()

        plt.text(
            column,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
        )

plt.tight_layout()
plt.show()

# %% CODE 10
confusion_matrix = torch.zeros(
    number_of_classes,
    number_of_classes,
    dtype=torch.int64,
)

for actual, predicted in zip(
    y_data,
    predicted_classes,
):
    confusion_matrix[actual, predicted] += 1

print("혼동행렬:")
print(confusion_matrix)

plt.figure(figsize=(5, 4))
plt.imshow(confusion_matrix.numpy())

plt.xticks(
    range(number_of_classes),
    class_names,
)

plt.yticks(
    range(number_of_classes),
    class_names,
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Training Confusion Matrix")

for row in range(number_of_classes):
    for column in range(number_of_classes):
        plt.text(
            column,
            row,
            int(confusion_matrix[row, column]),
            ha="center",
            va="center",
        )

plt.colorbar()
plt.tight_layout()
plt.show()

# %% CODE 11
new_x = torch.tensor(
    [
        [1.0, 5.0, 5.0, 5.0],
        [3.0, 2.0, 4.0, 4.0],
        [2.0, 1.0, 2.0, 2.0],
    ],
    dtype=torch.float32,
)

model.eval()

with torch.no_grad():
    new_logits = model(new_x)

    new_probabilities = torch.softmax(
        new_logits,
        dim=1,
    )

    new_predictions = torch.argmax(
        new_logits,
        dim=1,
    )

print("입력                  | 확률 [A, B, C]             | 예측")

for inputs, probabilities, predicted in zip(
    new_x,
    new_probabilities,
    new_predictions,
):
    probability_list = [
        round(value, 4)
        for value in probabilities.tolist()
    ]

    print(
        f"{str(inputs.tolist()):<21} | "
        f"{str(probability_list):<27} | "
        f"{class_names[predicted.item()]}"
    )
