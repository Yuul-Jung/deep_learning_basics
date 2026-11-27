# ch05_1_fashion_mnist_cnn_training_evaluation.py — 원본: lab_Pytorch/ch05_1_fashion_mnist_cnn_training_evaluation.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
from pathlib import Path

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
PNG_DIR = Path("png")

DATA_DIR.mkdir(exist_ok=True)

MODEL_DIR.mkdir(exist_ok=True)
PNG_DIR.mkdir(exist_ok=True)

print("데이터 폴더:", DATA_DIR)
print("모델 폴더  :", MODEL_DIR)
print("이미지 폴더:", PNG_DIR)

# %% CODE 02
import copy
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# %% CODE 03
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치:", device)

# %% CODE 04
BATCH_SIZE = 128

NUMBER_OF_EPOCHS = 10

LEARNING_RATE = 0.001

VALIDATION_SIZE = 5_000

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

class_names_ko = [
    "티셔츠",
    "바지",
    "풀오버",
    "원피스",
    "코트",
    "샌들",
    "셔츠",
    "운동화",
    "가방",
    "앵클부츠"
]

print("Batch Size:", BATCH_SIZE)

print("Epoch:", NUMBER_OF_EPOCHS)

print("Learning Rate:", LEARNING_RATE)

# %% CODE 05
transform = transforms.ToTensor()

full_train_dataset = datasets.FashionMNIST(
    root=str(DATA_DIR / "torchvision"),
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.FashionMNIST(
    root=str(DATA_DIR / "torchvision"),
    train=False,
    transform=transform,
    download=True
)

train_size = (
    len(full_train_dataset)
    - VALIDATION_SIZE
)

split_generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [train_size, VALIDATION_SIZE],
    generator=split_generator
)

loader_generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=loader_generator
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# %% CODE 06
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("학습 데이터:", len(train_dataset))

print("검증 데이터:", len(validation_dataset))

print("테스트 데이터:", len(test_dataset))

# %% CODE 07
sample_images, sample_labels = next(
    iter(train_loader)
)

sample_grid = make_grid(
    sample_images[:10],
    nrow=5,
    padding=2
)

plt.figure(figsize=(10, 5))

plt.imshow(
    sample_grid.permute(1, 2, 0)
)

plt.title("Fashion-MNIST Sample Images")

plt.axis("off")

plt.show()

print("배치 이미지 Shape:", sample_images.shape)

print("배치 정답 Shape:", sample_labels.shape)

print()

print("표시된 이미지의 실제 클래스")

for index in range(10):
    print(
        f"{index}: "
        f"{class_names_ko[sample_labels[index].item()]}"
    )

# %% CODE 08
def make_cnn():
    return nn.Sequential(
        nn.Conv2d(
            1,
            8,
            kernel_size=3,
            padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(
            kernel_size=2,
            stride=2
        ),

        nn.Conv2d(
            8,
            16,
            kernel_size=3,
            padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(
            kernel_size=2,
            stride=2
        ),

        nn.Flatten(start_dim=1),
        nn.Linear(
            16 * 7 * 7,
            10
        )
    )

torch.manual_seed(RANDOM_SEED)

model = make_cnn().to(device)

print(model)

# %% CODE 09
layer_names = [
    "Conv1",
    "ReLU1",
    "Pool1",
    "Conv2",
    "ReLU2",
    "Pool2",
    "Flatten",
    "Linear"
]

features = sample_images[:4].to(device)

print("Input   :", features.shape)

# %% CODE 10
for layer_name, layer in zip(
    layer_names,
    model
):
    features = layer(features)

    print(
        f"{layer_name:<8}:",
        features.shape
    )

# %% CODE 11
number_of_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print()

print(
    "CNN 학습 파라미터 수:",
    f"{number_of_parameters:,}개"
)

print(
    "이전 MLP 학습 파라미터 수:",
    "235,146개"
)

# %% CODE 12
loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# %% CODE 13
def train_one_epoch(
    model,
    data_loader
):
    model.train()

    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = loss_function(
            logits,
            labels
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

    return (
        loss_sum / total,
        correct / total
    )

# %% CODE 14
def evaluate_model(
    model,
    data_loader
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
                labels
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

    return (
        loss_sum / total,
        correct / total
    )

# %% CODE 15
history = {
    "train_loss": [],
    "train_accuracy": [],
    "validation_loss": [],
    "validation_accuracy": []
}

best_validation_accuracy = -1.0

best_validation_loss = float("inf")

best_epoch = 0

best_model_state = copy.deepcopy(
    model.state_dict()
)

# %% CODE 16
for epoch in range(
    1,
    NUMBER_OF_EPOCHS + 1
):
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader
    )

    validation_loss, validation_accuracy = (
        evaluate_model(
            model,
            validation_loader
        )
    )

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["validation_loss"].append(
        validation_loss
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

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

    print(
        f"Epoch {epoch:>2d}/{NUMBER_OF_EPOCHS} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Accuracy {train_accuracy:.4f} | "
        f"Validation Loss {validation_loss:.4f} | "
        f"Validation Accuracy {validation_accuracy:.4f}"
    )

# %% CODE 17
model.load_state_dict(
    best_model_state
)

print()

print("최고 검증 Epoch:", best_epoch)

print(
    "최고 검증 정확도:",
    f"{best_validation_accuracy:.4f}"
)

# %% CODE 18
epoch_numbers = range(
    1,
    NUMBER_OF_EPOCHS + 1
)

plt.figure(figsize=(8, 5))

plt.plot(
    epoch_numbers,
    history["train_loss"],
    marker="o",
    label="Train Loss"
)

plt.plot(
    epoch_numbers,
    history["validation_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Fashion-MNIST CNN Loss")

plt.grid(alpha=0.3)

plt.legend()

plt.show()

# %% CODE 19
plt.figure(figsize=(8, 5))

plt.plot(
    epoch_numbers,
    history["train_accuracy"],
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    epoch_numbers,
    history["validation_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Fashion-MNIST CNN Accuracy")

plt.ylim(0.7, 1.0)

plt.grid(alpha=0.3)

plt.legend()

plt.show()

# %% CODE 20
test_loss, test_accuracy = evaluate_model(
    model,
    test_loader
)

print(f"테스트 손실: {test_loss:.4f}")

print(f"테스트 정확도: {test_accuracy:.4f}")

print(
    f"맞힌 이미지 수: "
    f"{int(test_accuracy * len(test_dataset)):,}"
    f" / {len(test_dataset):,}"
)

# %% CODE 21
image_batches = []

label_batches = []

logit_batches = []

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        logits = model(
            images.to(device)
        ).cpu()

        image_batches.append(images)
        label_batches.append(labels)
        logit_batches.append(logits)

all_images = torch.cat(
    image_batches
)

all_labels = torch.cat(
    label_batches
)

all_logits = torch.cat(
    logit_batches
)

all_probabilities = torch.softmax(
    all_logits,
    dim=1
)

all_confidences, all_predictions = (
    all_probabilities.max(dim=1)
)

print("전체 이미지 Shape:", all_images.shape)

print("전체 예측 Shape:", all_predictions.shape)

# %% CODE 22
confusion_matrix = torch.zeros(
    10,
    10,
    dtype=torch.int64
)

# %% CODE 23
for actual, predicted in zip(
    all_labels,
    all_predictions
):
    confusion_matrix[
        actual,
        predicted
    ] += 1

# %% CODE 24
plt.figure(figsize=(8, 7))

plt.imshow(
    confusion_matrix.numpy()
)

plt.xticks(range(10))

plt.yticks(range(10))

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.title("Fashion-MNIST CNN Confusion Matrix")

for row in range(10):
    for column in range(10):
        plt.text(
            column,
            row,
            int(confusion_matrix[row, column]),
            ha="center",
            va="center",
            fontsize=8
        )

plt.colorbar()

plt.tight_layout()

plt.show()

# %% CODE 25
class_correct = confusion_matrix.diag()

class_total = confusion_matrix.sum(
    dim=1
)

class_accuracy = (
    class_correct.float()
    / class_total
)

for class_index in range(10):
    print(
        f"{class_index}: "
        f"{class_names_ko[class_index]:<6} | "
        f"{class_accuracy[class_index].item():.4f}"
    )

lowest_class = torch.argmin(
    class_accuracy
).item()

print()

print(
    "가장 낮은 정확도의 의류:",
    class_names_ko[lowest_class]
)

# %% CODE 26
plt.figure(figsize=(9, 5))

plt.bar(
    range(10),
    class_accuracy.numpy()
)

plt.xticks(
    range(10),
    class_names_ko,
    rotation=35
)

plt.ylim(0.5, 1.0)

plt.xlabel("Class")

plt.ylabel("Accuracy")

plt.title("CNN Accuracy by Fashion Class")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()

# %% CODE 27
wrong_confusion = confusion_matrix.clone()

wrong_confusion.fill_diagonal_(0)

flat_index = wrong_confusion.argmax().item()

confused_actual = flat_index // 10

confused_predicted = flat_index % 10

confused_count = wrong_confusion[
    confused_actual,
    confused_predicted
].item()

print(
    f"실제 {class_names_ko[confused_actual]} "
    f"→ 예측 {class_names_ko[confused_predicted]}: "
    f"{confused_count}개"
)

# %% CODE 28
wrong_mask = (
    all_predictions != all_labels
)

wrong_images = all_images[
    wrong_mask
]

wrong_labels = all_labels[
    wrong_mask
]

wrong_predictions = all_predictions[
    wrong_mask
]

wrong_confidences = all_confidences[
    wrong_mask
]

number_of_examples = min(
    12,
    len(wrong_images)
)

wrong_grid = make_grid(
    wrong_images[:number_of_examples],
    nrow=4,
    padding=2
)

plt.figure(figsize=(10, 8))

plt.imshow(
    wrong_grid.permute(1, 2, 0)
)

plt.title("Representative CNN Errors")

plt.axis("off")

plt.show()

print("대표 오답")

# %% CODE 29
for index in range(number_of_examples):
    actual_name = class_names_ko[
        wrong_labels[index].item()
    ]

    predicted_name = class_names_ko[
        wrong_predictions[index].item()
    ]

    print(
        f"{index:>2d} | "
        f"A:{actual_name:<6} | "
        f"P:{predicted_name:<6} | "
        f"C:{wrong_confidences[index].item():.3f}"
    )

# %% CODE 30
def shift_right(
    images,
    pixels=2
):
    shifted = torch.zeros_like(
        images
    )

    shifted[
        :,
        :,
        :,
        pixels:
    ] = images[
        :,
        :,
        :,
        :-pixels
    ]

    return shifted

# %% CODE 31
shifted_correct = 0

prediction_changed = 0

total = 0

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        shifted_images = shift_right(
            images,
            pixels=2
        )

        original_predictions = model(
            images
        ).argmax(dim=1)

        shifted_predictions = model(
            shifted_images
        ).argmax(dim=1)

        shifted_correct += (
            shifted_predictions == labels
        ).sum().item()

        prediction_changed += (
            shifted_predictions
            != original_predictions
        ).sum().item()

        total += labels.size(0)

shifted_accuracy = (
    shifted_correct / total
)

prediction_change_rate = (
    prediction_changed / total
)

accuracy_drop = (
    test_accuracy - shifted_accuracy
)

print(f"원본 정확도: {test_accuracy:.4f}")

# %% CODE 32
print(
    f"2픽셀 이동 정확도: "
    f"{shifted_accuracy:.4f}"
)

print(
    f"정확도 감소: "
    f"{accuracy_drop:.4f}"
)

print(
    f"예측 클래스 변경 비율: "
    f"{prediction_change_rate:.4f}"
)

# %% CODE 33
checkpoint_path = MODEL_DIR / "ch05_1_best_fashion_mnist_cnn.pt"

checkpoint = {
    "model_state_dict": model.state_dict(),
    "input_channels": 1,
    "feature_channels": [8, 16],
    "image_size": [1, 28, 28],
    "number_of_classes": 10,
    "class_names": class_names,
    "class_names_ko": class_names_ko,
    "best_epoch": best_epoch,
    "validation_loss": best_validation_loss,
    "validation_accuracy": best_validation_accuracy,
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
    "shifted_accuracy": shifted_accuracy
}

torch.save(
    checkpoint,
    checkpoint_path
)

loaded_checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

restored_model = make_cnn()

restored_model.load_state_dict(
    loaded_checkpoint[
        "model_state_dict"
    ]
)

restored_model.eval()

print("저장 파일:", checkpoint_path)

print(
    "저장된 검증 정확도:",
    f"{loaded_checkpoint['validation_accuracy']:.4f}"
)

print(
    "저장된 테스트 정확도:",
    f"{loaded_checkpoint['test_accuracy']:.4f}"
)
