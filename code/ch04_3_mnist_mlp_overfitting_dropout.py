# ch04_3_mnist_mlp_overfitting_dropout.py — 원본: lab_Pytorch/ch04_3_mnist_mlp_overfitting_dropout.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
from pathlib import Path

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
PNG_DIR = Path("png")

DATA_DIR.mkdir(exist_ok=True)

MODEL_DIR.mkdir(exist_ok=True)
PNG_DIR.mkdir(exist_ok=True)

print("데이터 폴더:", DATA_DIR)
print("모델 폴더 :", MODEL_DIR)
print("이미지 폴더:", PNG_DIR)

# %% CODE 02
import copy
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    random_split,
)
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# %% CODE 03
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치:", device)

# %% CODE 04
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,),
        ),
    ]
)

full_train_dataset = datasets.MNIST(
    root=str(DATA_DIR / "torchvision"),
    train=True,
    transform=transform,
    download=True,
)

test_dataset = datasets.MNIST(
    root=str(DATA_DIR / "torchvision"),
    train=False,
    transform=transform,
    download=True,
)

split_generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, validation_dataset, _ = random_split(
    full_train_dataset,
    [8_000, 2_000, 50_000],
    generator=split_generator,
)

print("실습용 훈련 데이터:", len(train_dataset))

print("검증 데이터:", len(validation_dataset))

print("테스트 데이터:", len(test_dataset))

# %% CODE 05
BATCH_SIZE = 128

NUMBER_OF_EPOCHS = 15

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

sample_images, sample_labels = next(
    iter(make_train_loader())
)

print("배치 이미지 shape:", sample_images.shape)

print("배치 정답 shape:", sample_labels.shape)

# %% CODE 06
def make_model(dropout_probability):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Dropout(dropout_probability),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(dropout_probability),
        nn.Linear(256, 10),
    )

torch.manual_seed(RANDOM_SEED)

initial_model = make_model(
    dropout_probability=0.0
)

initial_state = copy.deepcopy(
    initial_model.state_dict()
)

basic_model = make_model(
    dropout_probability=0.0
).to(device)

dropout_model = make_model(
    dropout_probability=0.3
).to(device)

basic_model.load_state_dict(initial_state)

dropout_model.load_state_dict(initial_state)

parameter_count = sum(
    parameter.numel()
    for parameter in basic_model.parameters()
)

print("기본 MLP 파라미터 수:", f"{parameter_count:,}")

print("Dropout MLP 파라미터 수:", f"{parameter_count:,}")

# %% CODE 07
loss_function = nn.CrossEntropyLoss()

# %% CODE 08
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

# %% CODE 09
def train_model(
    model,
    model_name,
):
    train_loader = make_train_loader()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    history = {
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }

    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0

    best_state = copy.deepcopy(
        model.state_dict()
    )

    print()
    print(f"[{model_name}] 학습 시작")

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

            optimizer.zero_grad()

            logits = model(images)

            loss = loss_function(
                logits,
                labels,
            )

            loss.backward()
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
            evaluate_model(
                model,
                validation_loader,
            )
        )

        history["train_loss"].append(
            train_loss
        )

        # (보완) 여기부터 함수 끝까지는 원본 PDF 쪽 아래로 넘쳐 잘린 부분이다.
        # 실습 4-2 run_experiment()와 같은 방식으로 채웠다(최고 검증 성능의 가중치를 저장·복원).
        history["validation_loss"].append(
            validation_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["validation_accuracy"].append(
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
            f"Train Accuracy {train_accuracy:.4f} | "
            f"Validation Loss {validation_loss:.4f} | "
            f"Validation Accuracy {validation_accuracy:.4f}"
        )

    # 최고 검증 성능의 가중치 복원
    model.load_state_dict(
        best_state
    )

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_accuracy": best_validation_accuracy,
    }

# %% CODE 10
basic_result = train_model(
    model=basic_model,
    model_name="기본 MLP",
)

# %% CODE 11
dropout_result = train_model(
    model=dropout_model,
    model_name="Dropout MLP",
)

# %% CODE 12
epoch_numbers = range(
    1,
    NUMBER_OF_EPOCHS + 1,
)

plt.figure(figsize=(8, 5))

plt.plot(
    epoch_numbers,
    basic_result["history"]["train_loss"],
    marker="o",
    label="Train Loss",
)

plt.plot(
    epoch_numbers,
    basic_result["history"][
        "validation_loss"
    ],
    marker="o",
    label="Validation Loss",
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Basic MLP Loss")

plt.grid(alpha=0.3)

plt.legend()

plt.show()

# %% CODE 13
plt.figure(figsize=(8, 5))

plt.plot(
    epoch_numbers,
    dropout_result["history"][
        "train_loss"
    ],
    marker="o",
    label="Train Loss",
)

plt.plot(
    epoch_numbers,
    dropout_result["history"][
        "validation_loss"
    ],
    marker="o",
    label="Validation Loss",
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Dropout MLP Loss")

plt.grid(alpha=0.3)

plt.legend()

plt.show()

# %% CODE 14
plt.figure(figsize=(8, 5))

plt.plot(
    epoch_numbers,
    basic_result["history"][
        "validation_accuracy"
    ],
    marker="o",
    label="Basic MLP",
)

plt.plot(
    epoch_numbers,
    dropout_result["history"][
        "validation_accuracy"
    ],
    marker="o",
    label="Dropout MLP",
)

plt.xlabel("Epoch")

plt.ylabel("Validation Accuracy")

plt.title("Validation Accuracy")

plt.ylim(0.8, 1.0)

plt.grid(alpha=0.3)

plt.legend()

plt.show()

# %% CODE 15
basic_gap = [
    train_accuracy - validation_accuracy
    for train_accuracy, validation_accuracy in zip(
        basic_result["history"]["train_accuracy"],
        basic_result["history"][
            "validation_accuracy"
        ],
    )
]

dropout_gap = [
    train_accuracy - validation_accuracy
    for train_accuracy, validation_accuracy in zip(
        dropout_result["history"][
            "train_accuracy"
        ],
        dropout_result["history"][
            "validation_accuracy"
        ],
    )
]

plt.figure(figsize=(8, 5))

plt.plot(
    epoch_numbers,
    basic_gap,
    marker="o",
    label="Basic MLP",
)

plt.plot(
    epoch_numbers,
    dropout_gap,
    marker="o",
    label="Dropout MLP",
)

plt.xlabel("Epoch")

plt.ylabel("Generalization Gap")

plt.title("Generalization Gap")

plt.grid(alpha=0.3)

plt.legend()

plt.show()

# %% CODE 16
def summarize_result(
    model_name,
    result,
):
    history = result["history"]

    final_train_accuracy = history[
        "train_accuracy"
    ][-1]

    final_validation_accuracy = history[
        "validation_accuracy"
    ][-1]

    final_gap = (
        final_train_accuracy
        - final_validation_accuracy
    )

    return {
        "model_name": model_name,
        "best_epoch": result["best_epoch"],
        "best_validation_loss": result[
            "best_validation_loss"
        ],
        "best_validation_accuracy": result[
            "best_validation_accuracy"
        ],
        "final_train_accuracy": (
            final_train_accuracy
        ),
        "final_validation_accuracy": (
            final_validation_accuracy
        ),
        "final_gap": final_gap,
    }

# %% CODE 17
summaries = [
    summarize_result(
        "기본 MLP",
        basic_result,
    ),
    summarize_result(
        "Dropout MLP",
        dropout_result,
    ),
]

# %% CODE 18
for summary in summaries:
    print()
    print(f"[{summary['model_name']}]")

    print(
        "최고 검증 Epoch:",
        summary["best_epoch"],
    )

    print(
        "최고 검증 손실:",
        f"{summary['best_validation_loss']:.4f}",
    )

    print(
        "최고 검증 정확도:",
        f"{summary['best_validation_accuracy']:.4f}",
    )

    print(
        "마지막 훈련 정확도:",
        f"{summary['final_train_accuracy']:.4f}",
    )

    print(
        "마지막 검증 정확도:",
        f"{summary['final_validation_accuracy']:.4f}",
    )

    print(
        "마지막 일반화 간격:",
        f"{summary['final_gap']:.4f}",
    )

# %% CODE 19
sample_images, _ = next(
    iter(validation_loader)
)

sample_image = sample_images[:1].to(device)

selected_dropout_model = dropout_result[
    "model"
]

selected_dropout_model.train()

train_output_1 = selected_dropout_model(
    sample_image
).detach().cpu()

train_output_2 = selected_dropout_model(
    sample_image
).detach().cpu()

selected_dropout_model.eval()

with torch.no_grad():
    eval_output_1 = selected_dropout_model(
        sample_image
    ).cpu()

    eval_output_2 = selected_dropout_model(
        sample_image
    ).cpu()

print(
    "train() 출력이 같은가?:",
    torch.allclose(
        train_output_1,
        train_output_2,
    ),
)

print(
    "eval() 출력이 같은가?:",
    torch.allclose(
        eval_output_1,
        eval_output_2,
    ),
)

# %% CODE 20
candidate_results = {
    "기본 MLP": basic_result,
    "Dropout MLP": dropout_result,
}

selected_model_name = max(
    candidate_results,
    key=lambda name: (
        candidate_results[name][
            "best_validation_accuracy"
        ],
        -candidate_results[name][
            "best_validation_loss"
        ],
    ),
)

selected_result = candidate_results[
    selected_model_name
]

selected_model = selected_result[
    "model"
]

selected_dropout_probability = (
    0.3
    if selected_model_name == "Dropout MLP"
    else 0.0
)

print("선택된 모델:", selected_model_name)

print(
    "선택 Epoch:",
    selected_result["best_epoch"],
)

print(
    "최고 검증 정확도:",
    f"{selected_result['best_validation_accuracy']:.4f}",
)

# %% CODE 21
test_loss, test_accuracy = evaluate_model(
    selected_model,
    test_loader,
)

print("선택된 모델:", selected_model_name)

print(f"테스트 손실: {test_loss:.4f}")

print(f"테스트 정확도: {test_accuracy:.4f}")

print(
    f"맞힌 이미지 수: "
    f"{int(test_accuracy * len(test_dataset)):,}"
    f" / {len(test_dataset):,}"
)

# %% CODE 22
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

# %% CODE 23
for index in range(5):
    actual = test_labels[index].item()
    predicted = test_predictions[index].item()

    confidence = test_probabilities[
        index,
        predicted,
    ].item()

    display_image = (
        test_images[index] * 0.3081
        + 0.1307
    )

    plt.figure(figsize=(2.5, 2.5))

    plt.imshow(
        display_image.squeeze().clamp(0, 1),
        cmap="gray",
    )

    plt.title(
        f"Actual: {actual} | "
        f"Predict: {predicted}\n"
        f"Probability: {confidence:.4f}"
    )

    plt.axis("off")
    plt.show()

# %% CODE 24
checkpoint_path = MODEL_DIR / "ch04_3_best_mnist_mlp.pt"

checkpoint = {
    "model_name": selected_model_name,
    "model_state_dict": (
        selected_model.state_dict()
    ),
    "dropout_probability": (
        selected_dropout_probability
    ),
    "hidden_sizes": [512, 256],
    "number_of_classes": 10,
    "normalization_mean": [0.1307],
    "normalization_std": [0.3081],
    "best_epoch": selected_result[
        "best_epoch"
    ],
    "validation_loss": selected_result[
        "best_validation_loss"
    ],
    "validation_accuracy": selected_result[
        "best_validation_accuracy"
    ],
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
}

torch.save(
    checkpoint,
    checkpoint_path,
)

loaded_checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True,
)

loaded_model = make_model(
    dropout_probability=loaded_checkpoint[
        "dropout_probability"
    ]
).to(device)

# %% CODE 25
loaded_model.load_state_dict(
    loaded_checkpoint[
        "model_state_dict"
    ]
)

loaded_model.eval()

print("저장 파일:", checkpoint_path)

print(
    "불러온 모델:",
    loaded_checkpoint["model_name"],
)

print(
    "저장된 검증 정확도:",
    f"{loaded_checkpoint['validation_accuracy']:.4f}",
)
