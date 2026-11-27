# ch05_extra_checks.py — 원본 실습 밖의 추가 확인 코드 (5장 해설에서 사용)

# %% CODE 01
# 강의 슬라이드 6·7: 4×4 입력과 3×3 필터의 합성곱 → 바이어스 → ReLU → 2×2 Max Pooling
import torch
import torch.nn.functional as F

x = torch.tensor([[1, 2, 3, 0],
                  [0, 1, 2, 3],
                  [3, 0, 1, 2],
                  [2, 3, 0, 1]], dtype=torch.float32)
w = torch.tensor([[2, 0, 1],
                  [0, 1, 2],
                  [1, 0, 2]], dtype=torch.float32)

# 손으로 하는 방법: 필터를 한 칸씩 옮기며 같은 자리끼리 곱해서 모두 더한다
manual = torch.zeros(2, 2)
for r in range(2):
    for c in range(2):
        manual[r, c] = (x[r:r + 3, c:c + 3] * w).sum()
print("손 계산 합성곱:\n", manual)

# PyTorch: 입력은 [N, C, H, W], 필터는 [출력 채널, 입력 채널, 높이, 너비]
torch_result = F.conv2d(x.view(1, 1, 4, 4), w.view(1, 1, 3, 3), bias=torch.tensor([3.0]))
print("F.conv2d + 바이어스 3:\n", torch_result[0, 0])
print("ReLU 뒤 :\n", F.relu(torch_result)[0, 0])
print("2×2 Max Pooling:", F.max_pool2d(F.relu(torch_result), 2).item())

# %% CODE 02
# 강의 슬라이드 5: 같은 2×2 영역을 세 가지 방법으로 줄이기
region = torch.tensor([[15.0, 16.0], [6.0, 15.0]])
print("max :", region.max().item(), "| min :", region.min().item(), "| average :", region.mean().item())

# %% CODE 03
# 강의 슬라이드 9: 제로 패딩 1칸을 두르면 4×4 입력에서 4×4 결과가 나온다
padded = F.conv2d(x.view(1, 1, 4, 4), w.view(1, 1, 3, 3), padding=1)
print("padding=1 결과 (4×4):\n", padded[0, 0])

# %% CODE 04
# 강의 슬라이드 22–24: 출력 크기 = floor((입력 − 필터 + 2×패딩) / 스트라이드) + 1
image = torch.zeros(1, 1, 28, 28)
for padding in (0, 1):
    for stride in (1, 2):
        out = torch.nn.Conv2d(1, 1, kernel_size=3, padding=padding, stride=stride)(image)
        formula = (28 - 3 + 2 * padding) // stride + 1
        print(f"padding={padding}, stride={stride} → 실제 {tuple(out.shape[-2:])}, 식 {formula}×{formula}")

# %% CODE 05
# 강의 슬라이드 18: 입력 채널이 여러 개면 채널별 합성곱 결과를 더해 출력 채널 하나를 만든다
torch.manual_seed(0)
two_channel = torch.randn(1, 2, 5, 5)                 # 입력 채널 2개
conv = torch.nn.Conv2d(2, 1, kernel_size=3, bias=False)  # 필터 하나 = [1, 2, 3, 3]
whole = conv(two_channel)
per_channel = sum(
    F.conv2d(two_channel[:, k:k + 1], conv.weight[:, k:k + 1]) for k in range(2)
)
print("필터 모양 [출력, 입력, 높이, 너비]:", list(conv.weight.shape))
print("채널별 결과를 더한 값과 같은가?:", torch.allclose(whole, per_channel, atol=1e-6))

# %% CODE 06
# 실습 5-1 CNN과 실습 4-5 MLP의 층별 파라미터 수
from torch import nn

cnn = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
    nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
    nn.Flatten(), nn.Linear(16 * 7 * 7, 10),
)
for name, parameter in cnn.named_parameters():
    print(f"CNN {name:<9} {str(list(parameter.shape)):<15} {parameter.numel():>7,}")
mlp_total = (784 * 256 + 256) + (256 * 128 + 128) + (128 * 10 + 10)
cnn_total = sum(p.numel() for p in cnn.parameters())
print(f"CNN 합계 {cnn_total:,} | MLP(실습 4-5) 합계 {mlp_total:,} | 비율 {mlp_total / cnn_total:.1f}배")

# %% CODE 07
# 실습 5-2 CODE 12·13 점검: '음수가 가장 많은 채널'은 이 이미지에서 전부 음수라 ReLU 뒤 통째로 0이 된다
# 실습 5-2의 모델·데이터·get_cnn_stages()를 그대로 쓰기 위해 그 파일의 CODE 01~07을 다시 실행한다
import contextlib, io, re
from pathlib import Path
import matplotlib.pyplot as plt

lab_source = Path("ch05_2_cnn_filter_feature_map_visualization.py").read_text(encoding="utf-8")
lab_blocks = dict(re.findall(r"^# %% CODE (\d\d)\n(.*?)(?=^# %% CODE |\Z)", lab_source, flags=re.M | re.S))
with contextlib.redirect_stdout(io.StringIO()):
    for number in ["01", "02", "03", "04", "05", "06", "07"]:
        exec(lab_blocks[number])

sample_image, _ = test_dataset[0]
with torch.no_grad():
    stages = get_cnn_stages(model, sample_image.unsqueeze(0).to(device))
conv1_raw = stages["conv1"][0].cpu()
negative_ratio = (conv1_raw < 0).float().mean(dim=(1, 2))
print("Conv1 채널별 음수 비율:", [round(v, 3) for v in negative_ratio.tolist()])

negative_sum = -conv1_raw.clamp(max=0).sum(dim=(1, 2))
positive_sum = conv1_raw.clamp(min=0).sum(dim=(1, 2))
mixed = torch.minimum(negative_sum, positive_sum).argmax().item()   # 음수 쪽과 양수 쪽 크기가 둘 다 큰 채널
before = conv1_raw[mixed]
after = torch.relu(before)
pooled = F.max_pool2d(after.view(1, 1, 28, 28), 2)[0, 0]
print(f"음수 합 {-negative_sum[mixed].item():.1f}, 양수 합 {positive_sum[mixed].item():.1f}인 채널: {mixed + 1}번 | ReLU 전 음수 {int((before < 0).sum())}칸"
      f" → ReLU 후 0인 칸 {int((after == 0).sum())}칸 | 풀링 뒤 크기 {tuple(pooled.shape)}")

limit = before.abs().max().item()                       # 0을 가운데(흰색)로 두고 음수는 파랑, 양수는 빨강
print(f"ReLU 전 값 범위: {before.min().item():.3f} ~ {before.max().item():.3f}")
plt.figure(figsize=(9, 3.2))
for k, (title, picture) in enumerate([("Before ReLU", before), ("After ReLU", after), ("After Pool (14, 14)", pooled)]):
    plt.subplot(1, 3, k + 1)
    plt.imshow(picture, cmap="bwr", vmin=-limit, vmax=limit)
    plt.title(f"Channel {mixed + 1}: {title}", fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.show()
