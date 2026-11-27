# ch02_extra_checks.py — 원본 실습 밖의 추가 확인 코드 (2장 해설에서 사용)

# %% CODE 01
# 강의(1/2m 비용)와 실습(1/m 비용)의 기울기 비교: W=5, 데이터 (1,1),(2,2),(3,3)
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 2.0, 3.0])
W = 5.0

grad_lab = (2 * (W * x - y) * x).mean()        # 실습 2-3: cost = 1/m Σ(Wx-y)^2 의 미분
grad_lecture = ((W * x - y) * x).mean()        # 강의: cost = 1/(2m) Σ(Wx-y)^2 의 미분
print("실습식 기울기 (2/m Σ):", grad_lab.item())
print("강의식 기울기 (1/m Σ):", grad_lecture.item())
print("비율:", (grad_lab / grad_lecture).item())

# %% CODE 02
# 학습률 경계: cost(W) = 14/3 (W-1)^2 이면 한 번 갈 때 (W-1) 이 (1 - α·28/3) 배가 된다
# |1 - α·28/3| < 1 이어야 줄어들므로 α < 6/28 ≈ 0.214
for alpha in [0.20, 0.21, 0.22, 0.25]:
    w = 5.0
    for _ in range(50):
        w = w - alpha * (28 / 3) * (w - 1)
    print(f"α={alpha:.2f} → 50번 뒤 W = {w:.4g}")

# %% CODE 03
# 강의 슬라이드의 모양 퀴즈 확인: [5,3]·[?,?]=[5,1], [n,3]·[?,?]=[n,2]
X = torch.randn(5, 3)
print("[5,3] @ [3,1] →", (X @ torch.randn(3, 1)).shape)
print("[5,3] @ [3,2] →", (X @ torch.randn(3, 2)).shape)
try:
    X @ torch.randn(2, 1)
except RuntimeError as error:
    print("[5,3] @ [2,1] → 오류:", str(error).split("\n")[0])
