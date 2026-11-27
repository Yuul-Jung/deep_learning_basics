# ch04_2_mnist_optimizer_comparison.py — 원본: lab_Pytorch/ch04_2_mnist_optimizer_comparison.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

split_generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [55_000, 5_000],
    generator=split_generator,
)

BATCH_SIZE = 64
NUMBER_OF_EPOCHS = 5

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

print("학습 데이터:", len(train_dataset))
print("검증 데이터:", len(validation_dataset))
print("테스트 데이터:", len(test_dataset))

# %% CODE 03
def make_train_loader():
    loader_generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    return DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
    )


sample_loader = make_train_loader()
sample_images, sample_labels = next(
    iter(sample_loader)
)

print("배치 이미지 shape:", sample_images.shape)
print("배치 정답 shape:", sample_labels.shape)

# %% CODE 04
def make_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


# 모든 모델이 같은 가중치에서 시작하도록 기준 상태 저장
torch.manual_seed(RANDOM_SEED)

initial_model = make_model()
initial_state = copy.deepcopy(
    initial_model.state_dict()
)

print(initial_model)

# %% CODE 05
loss_function = nn.CrossEntropyLoss()

optimizer_settings = {
    "SGD": {
        "learning_rate": 0.1,
        "momentum": 0.0,
    },
    "SGD + Momentum": {
        "learning_rate": 0.1,
        "momentum": 0.9,
    },
    "Adam": {
        "learning_rate": 0.001,
    },
}


def make_optimizer(
    optimizer_name,
    model,
):
    settings = optimizer_settings[
        optimizer_name
    ]

    if optimizer_name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=settings["learning_rate"],
        )

    return torch.optim.SGD(
        model.parameters(),
        lr=settings["learning_rate"],
        momentum=settings["momentum"],
    )


for name, settings in optimizer_settings.items():
    print(name, "→", settings)

# %% CODE 06
def evaluate_model(
    model,
    data_loader,
):
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

# %% CODE 07
def run_experiment(
    optimizer_name,
):
    model = make_model().to(device)

    # 같은 초기 가중치 적용
    model.load_state_dict(
        initial_state
    )

    train_loader = make_train_loader()

    optimizer = make_optimizer(
        optimizer_name,
        model,
    )

    train_loss_history = []
    validation_loss_history = []
    validation_accuracy_history = []

    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0

    best_state = copy.deepcopy(
        model.state_dict()
    )

    print()
    print(f"[{optimizer_name}] 학습 시작")

    for epoch in range(
        1,
        NUMBER_OF_EPOCHS + 1,
    ):
        model.train()

        train_loss_sum = 0.0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = loss_function(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            train_loss_sum += (
                loss.item()
                * images.size(0)
            )

            train_total += images.size(0)

        train_loss = (
            train_loss_sum
            / train_total
        )

        validation_loss, validation_accuracy = (
            evaluate_model(
                model,
                validation_loader,
            )
        )

        train_loss_history.append(
            train_loss
        )

        validation_loss_history.append(
            validation_loss
        )

        validation_accuracy_history.append(
            validation_accuracy
        )

        is_better = (
            validation_accuracy
            > best_validation_accuracy
            or (
                validation_accuracy
                == best_validation_accuracy
                and validation_loss
                < best_validation_loss
            )
        )

        if is_better:
            best_validation_accuracy = (
                validation_accuracy
            )

            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

        print(
            f"Epoch {epoch}/{NUMBER_OF_EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Validation Loss {validation_loss:.4f} | "
            f"Validation Accuracy {validation_accuracy:.4f}"
        )

    # 최고 검증 성능의 가중치 복원
    model.load_state_dict(
        best_state
    )

    return {
        "model": model,
        "train_loss": train_loss_history,
        "validation_loss": validation_loss_history,
        "validation_accuracy": validation_accuracy_history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_accuracy": best_validation_accuracy,
    }

# %% CODE 08
experiment_results = {}

for optimizer_name in optimizer_settings:
    experiment_results[
        optimizer_name
    ] = run_experiment(
        optimizer_name
    )

# %% CODE 09
epoch_numbers = range(
    1,
    NUMBER_OF_EPOCHS + 1,
)

plt.figure(figsize=(8, 5))

for optimizer_name, result in (
    experiment_results.items()
):
    plt.plot(
        epoch_numbers,
        result["validation_loss"],
        marker="o",
        label=optimizer_name,
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss by Optimizer")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# %% CODE 10
plt.figure(figsize=(8, 5))

for optimizer_name, result in (
    experiment_results.items()
):
    plt.plot(
        epoch_numbers,
        result["validation_accuracy"],
        marker="o",
        label=optimizer_name,
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy by Optimizer")
plt.ylim(0.8, 1.0)
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# %% CODE 11
print(
    "Optimizer         | Best Epoch | "
    "Best Val Loss | Best Val Accuracy"
)

print("-" * 69)

for optimizer_name, result in (
    experiment_results.items()
):
    print(
        f"{optimizer_name:<17} | "
        f"{result['best_epoch']:>10d} | "
        f"{result['best_validation_loss']:>13.4f} | "
        f"{result['best_validation_accuracy']:>17.4f}"
    )

# %% CODE 12
selected_optimizer_name = max(
    experiment_results,
    key=lambda name: (
        experiment_results[name][
            "best_validation_accuracy"
        ],
        -experiment_results[name][
            "best_validation_loss"
        ],
    ),
)

selected_result = experiment_results[
    selected_optimizer_name
]

selected_model = selected_result[
    "model"
]

print(
    "검증으로 선택한 Optimizer:",
    selected_optimizer_name,
)

print(
    "선택 Epoch:",
    selected_result["best_epoch"],
)

# %% CODE 13
test_loss, test_accuracy = evaluate_model(
    selected_model,
    test_loader,
)

print(
    "선택한 Optimizer:",
    selected_optimizer_name,
)

print(
    f"테스트 손실: {test_loss:.4f}"
)

print(
    f"테스트 정확도: {test_accuracy:.4f}"
)

print(
    f"맞힌 이미지 수: "
    f"{int(test_accuracy * len(test_dataset)):,}"
    f" / {len(test_dataset):,}"
)

# %% CODE 14
test_images, test_labels = next(
    iter(test_loader)
)

selected_model.eval()

with torch.no_grad():
    test_logits = selected_model(
        test_images.to(device)
    )

    test_probabilities = torch.softmax(
        test_logits,
        dim=1,
    ).cpu()

    test_predictions = test_logits.argmax(
        dim=1
    ).cpu()

for index in range(5):
    actual = test_labels[index].item()
    predicted = test_predictions[index].item()

    confidence = test_probabilities[
        index,
        predicted,
    ].item()

    plt.figure(figsize=(2.5, 2.5))

    plt.imshow(
        test_images[index].squeeze(),
        cmap="gray",
    )

    plt.title(
        f"Actual: {actual} | "
        f"Predict: {predicted}\n"
        f"Probability: {confidence:.4f}"
    )

    plt.axis("off")
    plt.show()
