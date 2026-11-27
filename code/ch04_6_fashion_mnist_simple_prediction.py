# ch04_6_fashion_mnist_simple_prediction.py — 원본: lab_Pytorch/ch04_6_fashion_mnist_simple_prediction.ipynb 의 코드 셀을 순서대로 옮김(이 실습은 PDF 없이 노트북만 있음)

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
from pathlib import Path
import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# %% CODE 03
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치:", device)

# %% CODE 04
checkpoint_path = MODEL_DIR / "ch04_5_best_fashion_mnist_mlp.pt"

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"{checkpoint_path.name} 파일이 없습니다. "
        "저장된 Fashion-MNIST MLP Checkpoint를 준비해 주세요."
    )

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

print("Checkpoint 불러오기 완료")

print(
    "저장된 검증 정확도:",
    f"{checkpoint['validation_accuracy']:.4f}"
)

print(
    "저장된 테스트 정확도:",
    f"{checkpoint['test_accuracy']:.4f}"
)

# %% CODE 05
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("모델 복원 완료")

# %% CODE 06
transform = transforms.ToTensor()

test_dataset = datasets.FashionMNIST(
    root=str(DATA_DIR / "torchvision"),
    train=False,
    transform=transform,
    download=True
)

class_names_ko = checkpoint[
    "class_names_ko"
]

print("테스트 데이터:", len(test_dataset))

print("의류 클래스 수:", len(class_names_ko))

# %% CODE 07
TEST_IMAGE_INDEX = 0

image, true_label = test_dataset[
    TEST_IMAGE_INDEX
]

print("선택한 이미지 번호:", TEST_IMAGE_INDEX)

print("이미지 shape:", image.shape)

print(
    "실제 의류:",
    class_names_ko[true_label]
)

# %% CODE 08
input_image = image.unsqueeze(
    0
).to(device)

print(
    "모델 입력 shape:",
    input_image.shape
)

# %% CODE 09
with torch.no_grad():
    logits = model(
        input_image
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predicted_label = logits.argmax(
        dim=1
    ).item()

    predicted_probability = probabilities[
        0,
        predicted_label
    ].item()

print("로짓 shape:", logits.shape)

print(
    "실제 의류:",
    class_names_ko[true_label]
)

print(
    "예측 의류:",
    class_names_ko[predicted_label]
)

print(
    f"예측 확률: "
    f"{predicted_probability * 100:.2f}%"
)

print(
    "예측 결과:",
    "정답" if predicted_label == true_label else "오답"
)

# %% CODE 10
top_probabilities, top_labels = torch.topk(
    probabilities[0],
    k=3
)

print("상위 3개 예측")

for rank, (
    label,
    probability
) in enumerate(
    zip(
        top_labels,
        top_probabilities
    ),
    start=1
):
    label_index = label.item()

    print(
        f"{rank}위 | "
        f"{class_names_ko[label_index]} | "
        f"{probability.item() * 100:.2f}%"
    )

# %% CODE 11
plt.figure(figsize=(4, 4))

plt.imshow(
    image.squeeze(),
    cmap="gray"
)

plt.title(
    f"실제: {class_names_ko[true_label]}\n"
    f"예측: {class_names_ko[predicted_label]} "
    f"({predicted_probability * 100:.2f}%)"
)

plt.axis("off")

plt.show()

# %% CODE 12
top_names = [
    class_names_ko[label.item()]
    for label in top_labels
]

top_percentages = [
    probability.item() * 100
    for probability in top_probabilities
]

plt.figure(figsize=(7, 4))

plt.bar(
    top_names,
    top_percentages
)

plt.xlabel("Class")

plt.ylabel("Probability (%)")

plt.title("Top 3 Predictions")

plt.ylim(0, 100)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.show()

# %% CODE 13
images = torch.stack(
    [
        test_dataset[index][0]
        for index in range(8)
    ]
)

labels = torch.tensor(
    [
        test_dataset[index][1]
        for index in range(8)
    ]
)

with torch.no_grad():
    batch_logits = model(
        images.to(device)
    )

    batch_probabilities = torch.softmax(
        batch_logits,
        dim=1
    )

    batch_confidences, batch_predictions = (
        batch_probabilities.max(dim=1)
    )

batch_predictions = batch_predictions.cpu()

batch_confidences = batch_confidences.cpu()

# %% CODE 14
for index in range(8):
    actual_name = class_names_ko[
        labels[index].item()
    ]

    predicted_name = class_names_ko[
        batch_predictions[index].item()
    ]

    print(
        f"{index}번 | "
        f"실제 {actual_name:<6} | "
        f"예측 {predicted_name:<6} | "
        f"{batch_confidences[index].item() * 100:.2f}%"
    )
