# ch04_4_mnist_prediction_error_analysis.py — 원본: lab_Pytorch/ch04_4_mnist_prediction_error_analysis.pdf 의 CODE 블록을 순서대로 옮김

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
import io
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
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
checkpoint_path = MODEL_DIR / "ch04_3_best_mnist_mlp.pt"

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"{checkpoint_path.name} 파일이 없습니다. "
        "저장된 MNIST MLP Checkpoint를 준비해 주세요."
    )

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

print("저장된 모델:", checkpoint["model_name"])

print(
    "Dropout 비율:",
    checkpoint["dropout_probability"]
)

# %% CODE 05
def make_model(dropout_probability):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Dropout(dropout_probability),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(dropout_probability),
        nn.Linear(256, 10)
    )

model = make_model(
    checkpoint["dropout_probability"]
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

normalization_mean = checkpoint[
    "normalization_mean"
][0]

normalization_std = checkpoint[
    "normalization_std"
][0]

print(model)

print()

print("정규화 평균:", normalization_mean)

print("정규화 표준편차:", normalization_std)

# %% CODE 06
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (normalization_mean,),
            (normalization_std,)
        )
    ]
)

test_dataset = datasets.MNIST(
    root=str(DATA_DIR / "torchvision"),
    train=False,
    transform=test_transform,
    download=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)

print("테스트 데이터:", len(test_dataset))

# %% CODE 07
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
            images.cpu()
        )

        label_batches.append(
            labels.cpu()
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

print("이미지 shape:", all_images.shape)

print("정답 shape:", all_labels.shape)

print("로짓 shape:", all_logits.shape)

# %% CODE 08
correct_mask = (
    all_predictions == all_labels
)

wrong_mask = ~correct_mask

correct_count = correct_mask.sum().item()

wrong_count = wrong_mask.sum().item()

accuracy = (
    correct_count / len(all_labels)
)

error_rate = (
    wrong_count / len(all_labels)
)

print(f"전체 정확도: {accuracy:.4f}")

print(f"오답률: {error_rate:.4f}")

print(
    f"정답 이미지: {correct_count:,}개"
)

print(
    f"오답 이미지: {wrong_count:,}개"
)

if "test_accuracy" in checkpoint:
    print()
    print(
        "Checkpoint 저장 정확도:",
        f"{checkpoint['test_accuracy']:.4f}"
    )

# %% CODE 09
sample_index = 0

sample_image = all_images[
    sample_index
]

sample_label = all_labels[
    sample_index
].item()

sample_probabilities = all_probabilities[
    sample_index
]

top_probabilities, top_classes = torch.topk(
    sample_probabilities,
    k=3
)

display_image = (
    sample_image * normalization_std
    + normalization_mean
).clamp(0, 1)

plt.figure(figsize=(3, 3))

plt.imshow(
    display_image.squeeze(),
    cmap="gray"
)

plt.title(
    f"Actual: {sample_label} | "
    f"Predicted: {top_classes[0].item()}"
)

plt.axis("off")

plt.show()

print("상위 3개 예측")

# %% CODE 10
for class_number, probability in zip(
    top_classes,
    top_probabilities
):
    print(
        f"숫자 {class_number.item()}: "
        f"{probability.item():.4f}"
    )

# %% CODE 11
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

print("오답 이미지 shape:", wrong_images.shape)

# %% CODE 12
number_of_examples = min(
    12,
    len(wrong_images)
)

plt.figure(figsize=(10, 8))

for index in range(number_of_examples):
    display_image = (
        wrong_images[index]
        * normalization_std
        + normalization_mean
    ).clamp(0, 1)

    plt.subplot(3, 4, index + 1)

    plt.imshow(
        display_image.squeeze(),
        cmap="gray"
    )

    plt.title(
        f"A:{wrong_labels[index].item()} "
        f"P:{wrong_predictions[index].item()}\n"
        f"C:{wrong_confidences[index].item():.3f}"
    )

    plt.axis("off")

plt.tight_layout()

plt.show()

# %% CODE 13
confusion_matrix = torch.zeros(
    10,
    10,
    dtype=torch.int64
)

# %% CODE 14
for actual, predicted in zip(
    all_labels,
    all_predictions
):
    confusion_matrix[
        actual,
        predicted
    ] += 1

# %% CODE 15
plt.figure(figsize=(8, 7))

plt.imshow(
    confusion_matrix.numpy()
)

plt.xticks(range(10))

plt.yticks(range(10))

plt.xlabel("Predicted Digit")

plt.ylabel("Actual Digit")

plt.title("MNIST Confusion Matrix")

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

# %% CODE 16
class_correct = confusion_matrix.diag()

class_total = confusion_matrix.sum(
    dim=1
)

class_accuracy = (
    class_correct.float()
    / class_total
)

for digit in range(10):
    print(
        f"숫자 {digit} | "
        f"정확도 {class_accuracy[digit].item():.4f} | "
        f"{class_correct[digit].item()}"
        f"/{class_total[digit].item()}"
    )

lowest_class = torch.argmin(
    class_accuracy
).item()

highest_class = torch.argmax(
    class_accuracy
).item()

print()

print("가장 낮은 정확도의 숫자:", lowest_class)

print("가장 높은 정확도의 숫자:", highest_class)

# %% CODE 17
plt.figure(figsize=(8, 5))

plt.bar(
    range(10),
    class_accuracy.numpy()
)

plt.xticks(range(10))

plt.ylim(0.8, 1.0)

plt.xlabel("Digit")

plt.ylabel("Accuracy")

plt.title("Accuracy by Digit")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.show()

# %% CODE 18
wrong_confusion = confusion_matrix.clone()

wrong_confusion.fill_diagonal_(0)

top_counts, top_positions = torch.topk(
    wrong_confusion.flatten(),
    k=5
)

print("가장 자주 발생한 오분류 관계")

for count, position in zip(
    top_counts,
    top_positions
):
    actual = (
        position // 10
    ).item()

    predicted = (
        position % 10
    ).item()

    print(
        f"실제 {actual} → 예측 {predicted}: "
        f"{count.item()}개"
    )

# %% CODE 19
most_confused_position = top_positions[0]

confused_actual = (
    most_confused_position // 10
).item()

confused_predicted = (
    most_confused_position % 10
).item()

pair_mask = (
    (all_labels == confused_actual)
    & (all_predictions == confused_predicted)
)

pair_images = all_images[
    pair_mask
]

pair_confidences = all_confidences[
    pair_mask
]

number_of_pair_examples = min(
    8,
    len(pair_images)
)

print(
    f"실제 {confused_actual} → "
    f"예측 {confused_predicted}"
)

plt.figure(figsize=(10, 4))

for index in range(
    number_of_pair_examples
):
    display_image = (
        pair_images[index]
        * normalization_std
        + normalization_mean
    ).clamp(0, 1)

    plt.subplot(
        2,
        4,
        index + 1
    )

    plt.imshow(
        display_image.squeeze(),
        cmap="gray"
    )

    plt.title(
        f"C:{pair_confidences[index].item():.3f}"
    )

    plt.axis("off")

plt.tight_layout()

plt.show()

# %% CODE 20
correct_confidences = all_confidences[
    correct_mask
]

print(
    "정답 평균 확신도:",
    f"{correct_confidences.mean().item():.4f}"
)

print(
    "오답 평균 확신도:",
    f"{wrong_confidences.mean().item():.4f}"
)

# %% CODE 21
high_confidence_values, high_confidence_indices = (
    torch.topk(
        wrong_confidences,
        k=min(8, len(wrong_confidences))
    )
)

plt.figure(figsize=(10, 4))

for plot_index, wrong_index in enumerate(
    high_confidence_indices
):
    display_image = (
        wrong_images[wrong_index]
        * normalization_std
        + normalization_mean
    ).clamp(0, 1)

    plt.subplot(
        2,
        4,
        plot_index + 1
    )

    plt.imshow(
        display_image.squeeze(),
        cmap="gray"
    )

    plt.title(
        f"A:{wrong_labels[wrong_index].item()} "
        f"P:{wrong_predictions[wrong_index].item()}\n"
        f"C:{high_confidence_values[plot_index].item():.3f}"
    )

    plt.axis("off")

plt.tight_layout()

plt.show()

# %% CODE 22
def preprocess_digit_image(image):
    grayscale = image.convert("L")

    image_array = np.array(
        grayscale
    )

    # 흰 배경이면 MNIST처럼 반전
    if image_array.mean() > 127:
        image_array = 255 - image_array

    # 약한 배경 잡음 제거
    image_array[
        image_array < 30
    ] = 0

    foreground_y, foreground_x = np.where(
        image_array > 0
    )

    if len(foreground_x) == 0:
        raise ValueError(
            "숫자 영역을 찾지 못했습니다."
        )

    left = foreground_x.min()
    right = foreground_x.max() + 1
    top = foreground_y.min()
    bottom = foreground_y.max() + 1

    cropped = Image.fromarray(
        image_array[
            top:bottom,
            left:right
        ]
    )

    scale = 20 / max(
        cropped.size
    )

    resized_size = (
        max(1, round(cropped.width * scale)),
        max(1, round(cropped.height * scale))
    )

    resized = cropped.resize(
        resized_size,
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "L",
        (28, 28),
        0
    )

    paste_position = (
        (28 - resized.width) // 2,
        (28 - resized.height) // 2
    )

    canvas.paste(
        resized,
        paste_position
    )

    tensor = transforms.ToTensor()(
        canvas
    )

    normalized_tensor = transforms.Normalize(
        (normalization_mean,),
        (normalization_std,)
    )(tensor)

    return canvas, normalized_tensor.unsqueeze(0)

# %% CODE 23
sample_tensor, sample_label = test_dataset[0]

original_image = transforms.ToPILImage()(
    sample_tensor
)

print("예제 정답:", sample_label)

processed_image, external_input = (
    preprocess_digit_image(
        original_image
    )
)

plt.figure(figsize=(3, 3))

plt.imshow(
    processed_image,
    cmap="gray"
)

plt.title("MNIST-like 28×28 Input")

plt.axis("off")

plt.show()

# %% CODE 24
model.eval()

with torch.no_grad():
    external_logits = model(
        external_input.to(device)
    )

    external_probabilities = torch.softmax(
        external_logits,
        dim=1
    ).cpu().squeeze(0)

top_probabilities, top_classes = torch.topk(
    external_probabilities,
    k=3
)

print("직접 그린 숫자의 상위 3개 예측")

for class_number, probability in zip(
    top_classes,
    top_probabilities
):
    print(
        f"숫자 {class_number.item()}: "
        f"{probability.item():.4f}"
    )
