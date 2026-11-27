# ch03_1_logistic_classification.py — 원본: lab_Pytorch/ch03_1_logistic_classification.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
import torch
from torch import nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

x_data = torch.tensor(
    [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [4.0, 3.0],
        [5.0, 3.0],
        [6.0, 2.0],
    ],
    dtype=torch.float32,
)

y_data = torch.tensor(
    [[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]],
    dtype=torch.float32,
)

print("입력 shape:", x_data.shape)
print("목표값 shape:", y_data.shape)
print()
print("입력 데이터:")
print(x_data)
print()
print("목표값:")
print(y_data)

# %% CODE 02
sample_logits = torch.tensor([-5.0, 0.0, 5.0])
sample_probabilities = torch.sigmoid(sample_logits)

for logit, probability in zip(
    sample_logits,
    sample_probabilities,
):
    print(
        f"로짓 {logit.item():>4.1f} "
        f"→ 확률 {probability.item():.4f}"
    )

# %% CODE 03
logit_values = torch.linspace(-8, 8, 200)
probability_values = torch.sigmoid(logit_values)

plt.figure(figsize=(8, 5))
plt.plot(logit_values.numpy(),
    probability_values.numpy(),)
plt.axhline(0.5, linestyle="--",
    label="Threshold = 0.5",)
plt.axvline(0, linestyle=":",)
plt.xlabel("Logit z")
plt.ylabel("Probability")
plt.title("Sigmoid Function")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# %% CODE 04
threshold = 0.5

sample_classes = (
    sample_probabilities >= threshold
).float()

for probability, predicted_class in zip(
    sample_probabilities,
    sample_classes,
):
    print(
        f"확률 {probability.item():.4f} "
        f"→ 클래스 {int(predicted_class.item())}"
    )

# %% CODE 05
bce = nn.BCELoss()

target_one = torch.tensor([[1.0]])

good_probability = torch.tensor([[0.9]])
wrong_probability = torch.tensor([[0.1]])

good_loss = bce(
    good_probability,
    target_one,
)

wrong_loss = bce(
    wrong_probability,
    target_one,
)

print("정답 1, 예측 확률 0.9의 Loss:", good_loss.item())
print("정답 1, 예측 확률 0.1의 Loss:", wrong_loss.item())

# %% CODE 06
torch.manual_seed(42)

model = nn.Linear(2, 1)

print(model)
print("가중치 shape:", model.weight.shape)
print("편향 shape:", model.bias.shape)

# %% CODE 07
loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
)

epochs = 2000
loss_history = []

model.train()

for epoch in range(epochs):
    # 1. 순전파: 로짓 계산
    logits = model(x_data)

    # 2. BCE 손실 계산
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

# %% CODE 08
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title("Training Loss")
plt.grid(alpha=0.3)
plt.show()

# %% CODE 09
model.eval()

with torch.no_grad():
    logits = model(x_data)
    probabilities = torch.sigmoid(logits)

    predicted_classes = (
        probabilities >= threshold
    ).float()

    accuracy = (
        predicted_classes == y_data
    ).float().mean()

print("학습된 가중치:", model.weight.detach())
print("학습된 편향:", model.bias.detach())
print(f"정확도: {accuracy.item():.4f}")

print()
print("입력             확률      예측  실제")

for x, probability, predicted, actual in zip(
    x_data,
    probabilities,
    predicted_classes,
    y_data,
):
    print(
        f"{x.tolist()}  "
        f"{probability.item():.4f}    "
        f"{int(predicted.item())}     "
        f"{int(actual.item())}"
    )

# %% CODE 10
actual_bool = y_data.bool()
predicted_bool = predicted_classes.bool()

tp = torch.sum(
    predicted_bool & actual_bool
).item()

tn = torch.sum(
    (~predicted_bool) & (~actual_bool)
).item()

fp = torch.sum(
    predicted_bool & (~actual_bool)
).item()

fn = torch.sum(
    (~predicted_bool) & actual_bool
).item()

print("TP:", tp)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)

# %% CODE 11
weight_1 = model.weight.detach()[0, 0]
weight_2 = model.weight.detach()[0, 1]
bias = model.bias.detach()[0]

x1_line = torch.linspace(0, 7, 200)

x2_line = -(
    weight_1 * x1_line + bias
) / weight_2

class_0 = y_data[:, 0] == 0
class_1 = y_data[:, 0] == 1

plt.figure(figsize=(8, 6))

plt.scatter(
    x_data[class_0, 0].numpy(),
    x_data[class_0, 1].numpy(),
    label="Class 0",
)

plt.scatter(
    x_data[class_1, 0].numpy(),
    x_data[class_1, 1].numpy(),
    label="Class 1",
)

plt.plot(
    x1_line.numpy(),
    x2_line.numpy(),
    linestyle="--",
    label="Decision Boundary",
)

plt.xlim(0, 7)
plt.ylim(0, 4)
plt.xlabel("Feature x1")
plt.ylabel("Feature x2")
plt.title("Logistic Classification")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# %% CODE 12
new_x = torch.tensor(
    [
        [2.0, 1.0],
        [4.0, 2.0],
        [6.0, 4.0],
    ],
    dtype=torch.float32,
)

model.eval()

with torch.no_grad():
    new_logits = model(new_x)

    new_probabilities = torch.sigmoid(
        new_logits
    )

    new_classes = (
        new_probabilities >= threshold
    ).float()

print("입력             로짓       확률      예측")

for x, logit, probability, predicted in zip(
    new_x,
    new_logits,
    new_probabilities,
    new_classes,
):
    print(
        f"{x.tolist()}  "
        f"{logit.item():8.4f}  "
        f"{probability.item():.4f}    "
        f"{int(predicted.item())}"
    )
