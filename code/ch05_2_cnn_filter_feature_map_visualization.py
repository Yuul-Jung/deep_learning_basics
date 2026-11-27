# ch05_2_cnn_filter_feature_map_visualization.py — 원본: lab_Pytorch/ch05_2_cnn_filter_feature_map_visualization.pdf 의 CODE 블록을 순서대로 옮김

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
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# %% CODE 03
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치:", device)

# %% CODE 04
checkpoint_path = MODEL_DIR / "ch05_1_best_fashion_mnist_cnn.pt"

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"{checkpoint_path.name} 파일이 없습니다. "
        "저장된 Fashion-MNIST CNN Checkpoint를 준비해 주세요."
    )

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

print("Checkpoint 불러오기 완료")

print(
    "저장된 테스트 정확도:",
    f"{checkpoint['test_accuracy']:.4f}"
)

# %% CODE 05
def make_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Flatten(start_dim=1),
        nn.Linear(16 * 7 * 7, 10),
    )

model = make_cnn().to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

class_names_ko = checkpoint[
    "class_names_ko"
]

print("CNN 복원 완료")

# %% CODE 06
transform = transforms.ToTensor()

test_dataset = datasets.FashionMNIST(
    root=str(DATA_DIR / "torchvision"),
    train=False,
    transform=transform,
    download=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)

print("테스트 데이터:", len(test_dataset))

# %% CODE 07
def get_cnn_stages(model, images):
    conv1 = model[0](images)
    relu1 = model[1](conv1)
    pool1 = model[2](relu1)

    conv2 = model[3](pool1)
    relu2 = model[4](conv2)
    pool2 = model[5](relu2)

    flattened = model[6](pool2)
    logits = model[7](flattened)

    return {
        "conv1": conv1,
        "relu1": relu1,
        "pool1": pool1,
        "conv2": conv2,
        "relu2": relu2,
        "pool2": pool2,
        "logits": logits,
    }

# %% CODE 08
SAMPLE_INDEX = 0

sample_image, sample_label = test_dataset[
    SAMPLE_INDEX
]

sample_input = sample_image.unsqueeze(
    0
).to(device)

with torch.no_grad():
    sample_stages = get_cnn_stages(
        model,
        sample_input
    )

sample_prediction = sample_stages[
    "logits"
].argmax(dim=1).item()

plt.figure(figsize=(3, 3))

plt.imshow(
    sample_image.squeeze(),
    cmap="gray"
)

plt.title(
    f"실제: {class_names_ko[sample_label]}\n"
    f"예측: {class_names_ko[sample_prediction]}"
)

plt.axis("off")

plt.show()

# %% CODE 09
print("입력  :", sample_input.shape)

for stage_name, stage_value in sample_stages.items():
    print(
        f"{stage_name:<7}:",
        stage_value.shape
    )

# %% CODE 10
conv1_filters = model[0].weight.detach().cpu()

print(
    "Conv1 필터 Shape:",
    conv1_filters.shape
)

plt.figure(figsize=(10, 5))

for index in range(8):
    plt.subplot(2, 4, index + 1)

    plt.imshow(
        conv1_filters[index, 0],
        cmap="gray"
    )

    plt.title(
        f"Filter {index + 1}"
    )

    plt.axis("off")

plt.suptitle("Learned Conv1 Filters")

plt.tight_layout()

plt.show()

# %% CODE 11
conv1_maps = sample_stages[
    "relu1"
].detach().cpu()

plt.figure(figsize=(10, 5))

for index in range(8):
    plt.subplot(2, 4, index + 1)

    plt.imshow(
        conv1_maps[0, index],
        cmap="gray"
    )

    plt.title(
        f"Map {index + 1}"
    )

    plt.axis("off")

plt.suptitle("Conv1 Feature Maps")

plt.tight_layout()

plt.show()

print(
    "Conv1 특징맵 Shape:",
    conv1_maps.shape
)

# %% CODE 12
conv1_raw = sample_stages[
    "conv1"
].detach().cpu()

negative_ratio = (
    conv1_raw[0] < 0
).float().mean(dim=(1, 2))

selected_channel = negative_ratio.argmax().item()

before_relu = conv1_raw[
    0,
    selected_channel
]

after_relu = conv1_maps[
    0,
    selected_channel
]

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)

plt.imshow(before_relu, cmap="gray")

plt.title("Before ReLU")

plt.axis("off")

plt.subplot(1, 2, 2)

plt.imshow(after_relu, cmap="gray")

plt.title("After ReLU")

plt.axis("off")

plt.tight_layout()

plt.show()

print(
    "선택한 채널:",
    selected_channel + 1
)

print(
    "ReLU 전 음수 개수:",
    int((before_relu < 0).sum())
)

print(
    "ReLU 후 음수 개수:",
    int((after_relu < 0).sum())
)

# %% CODE 13
before_pool = sample_stages[
    "relu1"
].detach().cpu()[0, selected_channel]

after_pool = sample_stages[
    "pool1"
].detach().cpu()[0, selected_channel]

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)

plt.imshow(before_pool, cmap="gray")

plt.title(
    f"Before Pool {tuple(before_pool.shape)}"
)

plt.axis("off")

plt.subplot(1, 2, 2)

plt.imshow(after_pool, cmap="gray")

plt.title(
    f"After Pool {tuple(after_pool.shape)}"
)

plt.axis("off")

plt.tight_layout()

plt.show()

# %% CODE 14
conv2_maps = sample_stages[
    "relu2"
].detach().cpu()

plt.figure(figsize=(10, 10))

for index in range(16):
    plt.subplot(4, 4, index + 1)

    plt.imshow(
        conv2_maps[0, index],
        cmap="gray"
    )

    plt.title(
        f"Map {index + 1}"
    )

    plt.axis("off")

plt.suptitle("Conv2 Feature Maps")

plt.tight_layout()

plt.show()

print(
    "Conv2 특징맵 Shape:",
    conv2_maps.shape
)

# %% CODE 15
conv2_scores = conv2_maps[0].mean(
    dim=(1, 2)
)

strong_channel = conv2_scores.argmax().item()

before_pool2 = conv2_maps[
    0,
    strong_channel
]

after_pool2 = sample_stages[
    "pool2"
].detach().cpu()[
    0,
    strong_channel
]

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)

plt.imshow(before_pool2, cmap="gray")

plt.title(
    f"Before Pool2 {tuple(before_pool2.shape)}"
)

plt.axis("off")

plt.subplot(1, 2, 2)

plt.imshow(after_pool2, cmap="gray")

plt.title(
    f"After Pool2 {tuple(after_pool2.shape)}"
)

plt.axis("off")

plt.tight_layout()

plt.show()

# %% CODE 16
correct_example = None

wrong_example = None

model.eval()

# %% CODE 17
with torch.no_grad():
    for images, labels in test_loader:
        logits = model(
            images.to(device)
        )

        predictions = logits.argmax(
            dim=1
        ).cpu()

        for image, label, prediction in zip(
            images,
            labels,
            predictions
        ):
            label_value = label.item()
            prediction_value = prediction.item()

            if (
                correct_example is None
                and label_value == prediction_value
            ):
                correct_example = (
                    image,
                    label_value,
                    prediction_value
                )

            if (
                wrong_example is None
                and label_value != prediction_value
            ):
                wrong_example = (
                    image,
                    label_value,
                    prediction_value
                )

            if (
                correct_example is not None
                and wrong_example is not None
            ):
                break

        if (
            correct_example is not None
            and wrong_example is not None
        ):
            break

# %% CODE 18
print("정답・오답 사례 찾기 완료")

# %% CODE 19
def show_example_features(
    model,
    example,
    title
):
    image, true_label, predicted_label = example

    image_input = image.unsqueeze(
        0
    ).to(device)

    with torch.no_grad():
        stages = get_cnn_stages(
            model,
            image_input
        )

    feature_maps = stages[
        "relu2"
    ].detach().cpu()

    scores = feature_maps[0].mean(
        dim=(1, 2)
    )

    top_channels = torch.topk(
        scores,
        k=3
    ).indices

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 4, 1)
    plt.imshow(
        image.squeeze(),
        cmap="gray"
    )
    plt.title(
        f"{title}\n"
        f"A:{class_names_ko[true_label]}\n"
        f"P:{class_names_ko[predicted_label]}"
    )
    plt.axis("off")

    for plot_index, channel in enumerate(
        top_channels,
        start=2
    ):
        plt.subplot(1, 4, plot_index)

        plt.imshow(
            feature_maps[0, channel],
            cmap="gray"
        )

        plt.title(
            f"Conv2 Map {channel.item() + 1}"
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()

# %% CODE 20
show_example_features(
    model,
    correct_example,
    "Correct"
)

show_example_features(
    model,
    wrong_example,
    "Wrong"
)
