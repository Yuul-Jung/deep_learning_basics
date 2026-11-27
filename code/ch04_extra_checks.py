# ch04_extra_checks.py — 원본 실습 밖의 추가 확인 코드 (4장 해설에서 사용)

# %% CODE 01
# 은닉층과 ReLU가 왜 필요한가: XOR(배타적 논리합)은 직선 하나로 나눌 수 없다
import torch
from torch import nn

x_xor = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_xor = torch.tensor([0, 1, 1, 0])


def train_xor(model, epochs=3000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_function = nn.CrossEntropyLoss()
    for _ in range(epochs):
        loss = loss_function(model(x_xor), y_xor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predictions = model(x_xor).argmax(dim=1)
    return loss.item(), predictions.tolist()


torch.manual_seed(0)
models = {
    "선형 1층 (3장 방식)": nn.Linear(2, 2),
    "선형 2층, ReLU 없음": nn.Sequential(nn.Linear(2, 8), nn.Linear(8, 2)),
    "선형 2층 + ReLU (MLP)": nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)),
}
print("정답               :", y_xor.tolist())
for name, model in models.items():
    loss, predictions = train_xor(model)
    print(f"{name:<18}: 예측 {predictions}, 마지막 Loss {loss:.4f}")

# %% CODE 02
# ReLU 없이 Linear를 쌓으면 Linear 하나와 같다: W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)
torch.manual_seed(0)
first, second = nn.Linear(784, 128), nn.Linear(128, 10)
merged = nn.Linear(784, 10)
with torch.no_grad():
    merged.weight.copy_(second.weight @ first.weight)
    merged.bias.copy_(second.weight @ first.bias + second.bias)
    x = torch.randn(5, 784)
    difference = (second(first(x)) - merged(x)).abs().max().item()
print("두 층 출력과 합친 한 층 출력의 최대 차이:", f"{difference:.2e}")

# %% CODE 03
# 실습 4-1 모델의 층별 파라미터 수와 배치 개수
model = nn.Sequential(
    nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 10)
)
for name, parameter in model.named_parameters():
    print(f"{name:<9} {str(list(parameter.shape)):<11} {parameter.numel():>7,}")
print("합계", f"{sum(p.numel() for p in model.parameters()):>27,}")
print()
full_batches, last_batch = divmod(55_000, 64)
print(f"55,000 ÷ 64 = 꽉 찬 배치 {full_batches}개 + 마지막 배치 {last_batch}장 → Iteration {full_batches + 1}")

# %% CODE 04
# 실습 4-4 CODE 23·24 점검: 정규화된 테스트 이미지를 '직접 그린 그림'처럼 다시 전처리하면 배경이 회색이 된다
# 실습 4-4의 모델·전처리 함수를 그대로 쓰기 위해 그 파일의 CODE 01~06, 22를 이 이름공간에서 다시 실행한다
import contextlib, io, re
from pathlib import Path

lab_source = Path("ch04_4_mnist_prediction_error_analysis.py").read_text(encoding="utf-8")
lab_blocks = dict(re.findall(r"^# %% CODE (\d\d)\n(.*?)(?=^# %% CODE |\Z)", lab_source, flags=re.M | re.S))
with contextlib.redirect_stdout(io.StringIO()):          # 실습 블록의 출력은 숨긴다
    for number in ["01", "02", "03", "04", "05", "06", "22"]:
        exec(lab_blocks[number])

raw_dataset = datasets.MNIST(root=str(DATA_DIR / "torchvision"), train=False,
                             transform=transforms.ToTensor(), download=True)


def center_of_mass(array):
    rows, cols = np.indices(array.shape)
    weight = array.astype(np.float64)
    return (rows * weight).sum() / weight.sum(), (cols * weight).sum() / weight.sum()


panels = []


def check(name, pil_image):
    canvas, model_input = preprocess_digit_image(pil_image)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(model_input.to(device)), dim=1)[0]
    top = probabilities.argmax().item()
    row, col = center_of_mass(np.array(canvas))
    print(f"{name:<20} | 원본 픽셀 평균 {np.array(pil_image).mean():6.1f} | 전처리 뒤 무게중심 (행 {row:4.1f}, 열 {col:4.1f})"
          f" | 예측 {top} ({probabilities[top].item():.4f})")
    panels.append((name, pil_image, canvas, top))


normalized_tensor, label = test_dataset[0]              # 실습 4-4가 쓴 것: 정규화된 텐서
raw_tensor, _ = raw_dataset[0]                          # 정규화 전 0~1 텐서
row, col = center_of_mass(np.array(transforms.ToPILImage()(raw_tensor)))
print(f"정답: {label}, 원래 MNIST 이미지의 무게중심 (행 {row:4.1f}, 열 {col:4.1f})")
with torch.no_grad():
    direct = torch.softmax(model(normalized_tensor.unsqueeze(0).to(device)), dim=1)[0]
print(f"전처리 없이 바로 넣으면        → 예측 {direct.argmax().item()} ({direct.max().item():.4f})")
check("정규화된 텐서 (실습)", transforms.ToPILImage()(normalized_tensor))
check("0~1 원본 텐서", transforms.ToPILImage()(raw_tensor))

plt.figure(figsize=(8, 4.2))
for k, (name, before, after, top) in enumerate(panels):
    plt.subplot(2, 2, 2 * k + 1); plt.imshow(before, cmap="gray"); plt.axis("off")
    plt.title(["normalized tensor -> PIL", "0~1 tensor -> PIL"][k], fontsize=9)
    plt.subplot(2, 2, 2 * k + 2); plt.imshow(after, cmap="gray"); plt.axis("off")
    plt.title(f"after preprocess -> predict {top}", fontsize=9)
plt.tight_layout()
plt.show()
