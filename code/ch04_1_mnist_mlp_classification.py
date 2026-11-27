# ch04_1_mnist_mlp_classification.py — 원본: lab_Pytorch/ch04_1_mnist_mlp_classification.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
"cpu")

print("사용 장치:", device)

# %% CODE 02
transform = transforms.ToTensor()

full_train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True,
)

print("전체 훈련 데이터:", len(full_train_dataset))
print("테스트 데이터:", len(test_dataset))

# %% CODE 03
sample_image, sample_label = full_train_dataset[0]

print("이미지 shape:", sample_image.shape)
print("정답 숫자:", sample_label)
print(
    "픽셀값 범위:",
    sample_image.min().item(),
    "~",
    sample_image.max().item(),
)

plt.figure(figsize=(3, 3))
plt.imshow(
    sample_image.squeeze(),
    cmap="gray",
)
plt.title(f"Label: {sample_label}")
plt.axis("off")
plt.show()

# %% CODE 04
split_generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [55_000, 5_000],
    generator=split_generator,
)

print("실제 학습 데이터:", len(train_dataset))
print("검증 데이터:", len(validation_dataset))
print("테스트 데이터:", len(test_dataset))

# %% CODE 05
BATCH_SIZE = 64
NUMBER_OF_EPOCHS = 5

loader_generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=loader_generator,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

batch_images, batch_labels = next(
    iter(train_loader)
)

print("배치 이미지 shape:", batch_images.shape)
print("배치 정답 shape:", batch_labels.shape)

# %% CODE 06
iterations_per_epoch = math.ceil(
    len(train_dataset) / BATCH_SIZE
)

maximum_weight_updates = (
    iterations_per_epoch
    * NUMBER_OF_EPOCHS
)

print("에포크당 Iteration:", iterations_per_epoch)
print(
    f"{NUMBER_OF_EPOCHS} Epoch 최대 가중치 갱신:",
    maximum_weight_updates,
)

# %% CODE 07
torch.manual_seed(RANDOM_SEED)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10),
).to(device)

print(model)

# %% CODE 08
model.eval()

with torch.no_grad():
    sample_logits = model(
        batch_images[:4].to(device)
    )

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("모델 출력 shape:", sample_logits.shape)
print(
    "전체 학습 파라미터 수:",
    f"{total_parameters:,}",
)

# %% CODE 09
loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

print("손실함수:", loss_function)
print("Optimizer:", optimizer.__class__.__name__)

# %% CODE 10
def evaluate_model(data_loader):
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = loss_function(
                logits,
                labels,
            )

            predictions = logits.argmax(
                dim=1
            )

            loss_sum += (
                loss.item()
                * images.size(0)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = loss_sum / total
    accuracy = correct / total

    return average_loss, accuracy

# %% CODE 11
train_loss_history = []
train_accuracy_history = []

validation_loss_history = []
validation_accuracy_history = []

for epoch in range(
    1,
    NUMBER_OF_EPOCHS + 1,
):
    model.train()

    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 1. 이전 기울기 초기화
        optimizer.zero_grad()

        # 2. 순전파
        logits = model(images)

        # 3. 손실 계산
        loss = loss_function(
            logits,
            labels,
        )

        # 4. 역전파
        loss.backward()

        # 5. 가중치 갱신
        optimizer.step()

        predictions = logits.argmax(
            dim=1
        )

        loss_sum += (
            loss.item()
            * images.size(0)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = loss_sum / total
    train_accuracy = correct / total

    validation_loss, validation_accuracy = (
        evaluate_model(validation_loader)
    )

    train_loss_history.append(
        train_loss
    )

    train_accuracy_history.append(
        train_accuracy
    )

    validation_loss_history.append(
        validation_loss
    )

    validation_accuracy_history.append(
        validation_accuracy
    )

    print(
        f"Epoch {epoch}/{NUMBER_OF_EPOCHS} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Accuracy {train_accuracy:.4f} | "
        f"Validation Loss {validation_loss:.4f} | "
        f"Validation Accuracy {validation_accuracy:.4f}"
    )  # (보완) 이 줄과 위 줄은 원본 PDF 쪽 아래로 넘쳐 잘린 부분

# %% CODE 12
epoch_numbers = range(
    1,
    NUMBER_OF_EPOCHS + 1,
)

plt.figure(figsize=(8, 5))

plt.plot(
    epoch_numbers,
    train_loss_history,
    marker="o",
    label="Train Loss",
)

plt.plot(
    epoch_numbers,
    validation_loss_history,
    marker="o",
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# %% CODE 13
plt.figure(figsize=(8, 5))

plt.plot(
    epoch_numbers,
    train_accuracy_history,
    marker="o",
    label="Train Accuracy",
)

plt.plot(
    epoch_numbers,
    validation_accuracy_history,
    marker="o",
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.ylim(0.8, 1.0)
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# %% CODE 14
test_loss, test_accuracy = evaluate_model(
    test_loader
)

print(f"테스트 손실: {test_loss:.4f}")
print(f"테스트 정확도: {test_accuracy:.4f}")
print(
    f"맞힌 이미지 수: "
    f"{int(test_accuracy * len(test_dataset)):,}"
    f" / {len(test_dataset):,}"
)

# %% CODE 15
model.eval()

display_images, display_labels = next(
    iter(test_loader)
)

with torch.no_grad():
    display_logits = model(
        display_images.to(device)
    )

    display_probabilities = torch.softmax(
        display_logits,
        dim=1,
    ).cpu()

    display_predictions = display_logits.argmax(
        dim=1
    ).cpu()

for index in range(5):
    actual = display_labels[index].item()
    predicted = display_predictions[index].item()

    confidence = display_probabilities[
        index,
        predicted,
    ].item()

    plt.figure(figsize=(2.5, 2.5))
    plt.imshow(
        display_images[index].squeeze(),
        cmap="gray",
    )
    plt.title(
        f"Actual: {actual} | "
        f"Predict: {predicted}\\n"
        f"Probability: {confidence:.4f}"
    )
    plt.axis("off")
    plt.show()

# %% CODE 16
first_probabilities = (
    display_probabilities[0]
)

first_prediction = (
    display_predictions[0].item()
)

print(
    "첫 번째 이미지 실제 정답:",
    display_labels[0].item(),
)

print(
    "첫 번째 이미지 예측:",
    first_prediction,
)

print()
print("숫자별 확률:")

for digit, probability in enumerate(
    first_probabilities
):
    print(
        f"{digit}: "
        f"{probability.item():.4f}"
    )

print(
    "확률 합:",
    first_probabilities.sum().item(),
)
