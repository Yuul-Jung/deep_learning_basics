# ch03_extra_checks.py — 원본 실습 밖의 추가 확인 코드 (3장 해설에서 사용)

# %% CODE 01
# 강의 슬라이드 7: 합격/불합격을 선형회귀 + 기준값 0.5로 나누면 멀리 있는 점 하나에 기준이 밀린다
import torch
import matplotlib.pyplot as plt

hours = torch.tensor([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
passed = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])


def boundary_by_linear_regression(x, y):
    # 최소제곱 직선 y = Wx + b 를 식으로 바로 구하고, H(x) = 0.5 가 되는 x 를 돌려준다
    X = torch.stack([x, torch.ones_like(x)], dim=1)
    W, b = torch.linalg.lstsq(X, y.unsqueeze(1)).solution[:, 0]
    return W.item(), b.item(), ((0.5 - b) / W).item()


W1, b1, cut1 = boundary_by_linear_regression(hours, passed)
hours2 = torch.cat([hours, torch.tensor([30.0])])        # 30시간 공부한 합격자 한 명 추가
passed2 = torch.cat([passed, torch.tensor([1.0])])
W2, b2, cut2 = boundary_by_linear_regression(hours2, passed2)

print(f"점 6개   : H(x) = {W1:.4f}x {b1:+.4f}, 0.5가 되는 공부 시간 = {cut1:.2f}")
print(f"점 7개   : H(x) = {W2:.4f}x {b2:+.4f}, 0.5가 되는 공부 시간 = {cut2:.2f}")
print(f"5시간 공부한 합격자: 점 6개일 때 H(5) = {W1 * 5 + b1:.4f}, 점 7개일 때 H(5) = {W2 * 5 + b2:.4f}")

xs = torch.linspace(0, 31, 200)
plt.figure(figsize=(8, 4.5))
plt.scatter(hours2.numpy(), passed2.numpy(), label="Data (0 = fail, 1 = pass)")
plt.plot(xs.numpy(), (W1 * xs + b1).numpy(), label=f"6 points: cut = {cut1:.2f} h")
plt.plot(xs.numpy(), (W2 * xs + b2).numpy(), linestyle="--", label=f"7 points: cut = {cut2:.2f} h")
plt.axhline(0.5, color="gray", linewidth=1)
plt.ylim(-0.3, 1.6)
plt.xlabel("Study hours")
plt.ylabel("H(x)")
plt.title("Linear Regression for Pass/Fail")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# %% CODE 02
# 같은 데이터에 로지스틱 회귀(시그모이드 + BCE)를 쓰면 기준(확률 0.5)이 거의 움직이지 않는다
def boundary_by_logistic(x, y, epochs=5000, lr=0.5):
    torch.manual_seed(0)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_function = torch.nn.BCEWithLogitsLoss()
    x_scaled = ((x - 4.0) / 2.0).unsqueeze(1)     # 같은 기준(4시간, 2시간)으로 크기만 맞춤
    for _ in range(epochs):
        loss = loss_function(model(x_scaled), y.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    w, b = model.weight.item(), model.bias.item()
    return -b / w * 2.0 + 4.0                       # 로짓 0(확률 0.5)이 되는 공부 시간


print(f"점 6개   : 확률 0.5가 되는 공부 시간 = {boundary_by_logistic(hours, passed):.2f}")
print(f"점 7개   : 확률 0.5가 되는 공부 시간 = {boundary_by_logistic(hours2, passed2):.2f}")

# %% CODE 03
# BCE 손실 = 강의의 c(H(x), y). y=1이면 −log p, y=0이면 −log(1−p)
from torch import nn

p = torch.tensor([0.9, 0.5, 0.1])
print("p              :", [round(v, 4) for v in p.tolist()])
print("y=1: −log p     :", [round(v, 4) for v in (-torch.log(p)).tolist()])
print("y=0: −log(1−p)  :", [round(v, 4) for v in (-torch.log(1 - p)).tolist()])

# 시그모이드 → BCELoss 와 BCEWithLogitsLoss 비교: 보통은 같고, 로짓이 극단이면 차이가 난다
for z, y in [(2.0, 1.0), (-200.0, 1.0)]:
    logit = torch.tensor([[z]])
    target = torch.tensor([[y]])
    two_step = nn.BCELoss()(torch.sigmoid(logit), target).item()
    one_step = nn.BCEWithLogitsLoss()(logit, target).item()
    print(f"로짓 {z:7.1f}, 정답 {int(y)} → sigmoid+BCELoss = {two_step:.4f}, BCEWithLogitsLoss = {one_step:.4f}")

# %% CODE 04
# 강의 슬라이드 23·24·31의 예 [2.0, 1.0, 0.1]: 시그모이드와 소프트맥스, 교차엔트로피
scores = torch.tensor([2.0, 1.0, 0.1])
sig = torch.sigmoid(scores)
soft = torch.softmax(scores, dim=0)
print("시그모이드 :", [round(v, 4) for v in sig.tolist()], "합 =", round(sig.sum().item(), 4))
print("소프트맥스 :", [round(v, 4) for v in soft.tolist()], "합 =", round(soft.sum().item(), 4))
print()
print("정답 클래스 | 강의 값(0.7·0.2·0.1 사용) | 실제 소프트맥스 값 사용")
for k in range(3):
    lecture = -torch.log(torch.tensor([0.7, 0.2, 0.1])[k]).item()
    actual = -torch.log(soft[k]).item()
    print(f"     {k + 1}      |          {lecture:.3f}           |        {actual:.3f}")
