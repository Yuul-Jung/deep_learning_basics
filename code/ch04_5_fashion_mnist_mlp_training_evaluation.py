# ch04_5_fashion_mnist_mlp_training_evaluation.py — 원본: lab_Pytorch/ch04_5_fashion_mnist_mlp_training_evaluation.ipynb 의 코드 셀을 순서대로 옮김(이 실습은 PDF 없이 노트북만 있음)

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
import matplotlib.pyplot as plt

# %% CODE 03
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치:", device)

# %% CODE 04
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

split_generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [55_000, 5_000],
    generator=split_generator
)

print("학습 데이터:", len(train_dataset))

print("검증 데이터:", len(validation_dataset))

print("테스트 데이터:", len(test_dataset))

# %% CODE 06
example_images = [None] * 10

for image, label in full_train_dataset:
    if example_images[label] is None:
        example_images[label] = image

    if all(image is not None for image in example_images):
        break

plt.figure(figsize=(12, 5))

for class_index, image in enumerate(example_images):
    plt.subplot(2, 5, class_index + 1)

    plt.imshow(
        image.squeeze(),
        cmap="gray"
    )

    plt.title(
        f"{class_index}: {class_names_ko[class_index]}"
    )

    plt.axis("off")

plt.tight_layout()

plt.show()

# %% CODE 07
BATCH_SIZE = 128

NUMBER_OF_EPOCHS = 10

LEARNING_RATE = 0.001

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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

batch_images, batch_labels = next(
    iter(train_loader)
)

print("배치 이미지 shape:", batch_images.shape)

print("배치 정답 shape:", batch_labels.shape)

# %% CODE 08
torch.manual_seed(RANDOM_SEED)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

number_of_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(model)

print(
    "학습 파라미터 수:",
    f"{number_of_parameters:,}개"
)

# %% CODE 09
loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# %% CODE 10
def train_one_epoch(model, data_loader):
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

        predictions = logits.argmax(dim=1)

        loss_sum += (
            loss.item() * images.size(0)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        loss_sum / total,
        correct / total
    )

# %% CODE 11
def evaluate_model(model, data_loader):
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

            predictions = logits.argmax(dim=1)

            loss_sum += (
                loss.item() * images.size(0)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        loss_sum / total,
        correct / total
    )

# %% CODE 12
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

# %% CODE 13
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

# %% CODE 14
model.load_state_dict(
    best_model_state
)

print()

print("최고 검증 Epoch:", best_epoch)

print(
    "최고 검증 정확도:",
    f"{best_validation_accuracy:.4f}"
)

# %% CODE 15
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

plt.title("Fashion-MNIST MLP Loss")

plt.grid(alpha=0.3)

plt.legend()

plt.show()

# %% CODE 16
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

plt.title("Fashion-MNIST MLP Accuracy")

plt.ylim(0.7, 1.0)

plt.grid(alpha=0.3)

plt.legend()

plt.show()

# %% CODE 17
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

# %% CODE 18
image_batches = []

label_batches = []

logit_batches = []

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        logits = model(
            images.to(device)
        ).cpu()

        image_batches.append(
            images
        )

        label_batches.append(
            labels
        )

        logit_batches.append(
            logits
        )

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

print("전체 이미지 shape:", all_images.shape)

print("전체 예측 shape:", all_predictions.shape)

# %% CODE 19
confusion_matrix = torch.zeros(
    10,
    10,
    dtype=torch.int64
)

# %% CODE 20
for actual, predicted in zip(
    all_labels,
    all_predictions
):
    confusion_matrix[
        actual,
        predicted
    ] += 1

# %% CODE 21
plt.figure(figsize=(8, 7))

plt.imshow(
    confusion_matrix.numpy()
)

plt.xticks(
    range(10),
    range(10)
)

plt.yticks(
    range(10),
    range(10)
)

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.title("Fashion-MNIST MLP Confusion Matrix")

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

# %% CODE 22
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

# %% CODE 23
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

plt.title("Accuracy by Fashion Class")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()

# %% CODE 24
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

# %% CODE 25
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

plt.figure(figsize=(10, 8))

for index in range(number_of_examples):
    plt.subplot(
        3,
        4,
        index + 1
    )

    plt.imshow(
        wrong_images[index].squeeze(),
        cmap="gray"
    )

    actual_name = class_names_ko[
        wrong_labels[index].item()
    ]

    predicted_name = class_names_ko[
        wrong_predictions[index].item()
    ]

    plt.title(
        f"A:{actual_name} P:{predicted_name}\n"
        f"C:{wrong_confidences[index].item():.3f}",
        fontsize=9
    )

    plt.axis("off")

plt.tight_layout()

plt.show()

# %% CODE 26
def shift_right(images, pixels=2):
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

# %% CODE 27
original_image = all_images[:1]

shifted_image = shift_right(
    original_image,
    pixels=2
)

plt.figure(figsize=(3, 3))

plt.imshow(
    original_image[0].squeeze(),
    cmap="gray"
)

plt.title("Original")

plt.axis("off")

plt.show()

plt.figure(figsize=(3, 3))

plt.imshow(
    shifted_image[0].squeeze(),
    cmap="gray"
)

plt.title("Shifted Right by 2 Pixels")

plt.axis("off")

plt.show()

# %% CODE 28
model.eval()

shifted_correct = 0

prediction_changed = 0

total = 0

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

# %% CODE 29
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

# %% CODE 30
checkpoint_path = MODEL_DIR / "ch04_5_best_fashion_mnist_mlp.pt"

checkpoint = {
    "model_state_dict": model.state_dict(),
    "hidden_sizes": [256, 128],
    "number_of_classes": 10,
    "class_names": class_names,
    "class_names_ko": class_names_ko,
    "image_size": [1, 28, 28],
    "best_epoch": best_epoch,
    "validation_loss": best_validation_loss,
    "validation_accuracy": best_validation_accuracy,
    "test_loss": test_loss,
    "test_accuracy": test_accuracy
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

print("저장 파일:", checkpoint_path)

print(
    "저장된 검증 정확도:",
    f"{loaded_checkpoint['validation_accuracy']:.4f}"
)

print(
    "저장된 테스트 정확도:",
    f"{loaded_checkpoint['test_accuracy']:.4f}"
)
