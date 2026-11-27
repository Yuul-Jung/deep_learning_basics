# ch01_extra_checks.py — 원본 실습 밖의 추가 확인 코드 (1장 해설에서 사용)

# %% CODE 01
import numpy as np
import torch

# torch.tensor() 는 복사, torch.from_numpy() 와 .numpy() 는 메모리 공유
arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
t_copy = torch.tensor(arr)        # 새 메모리에 복사
t_share = torch.from_numpy(arr)   # arr 과 같은 메모리
arr[0] = 100.0
print("arr          :", arr)
print("torch.tensor :", t_copy)
print("from_numpy   :", t_share)

back = t_copy.numpy()             # 텐서 → 배열도 메모리 공유
t_copy[1] = -1.0
print("t_copy.numpy():", back)

# %% CODE 02
# 기본 자료형: NumPy 실수는 float64, PyTorch 실수는 float32
np_default = np.array([1.5, 2.5])
torch_default = torch.tensor([1.5, 2.5])
from_np = torch.from_numpy(np_default)
print("NumPy 기본       :", np_default.dtype)
print("PyTorch 기본     :", torch_default.dtype)
print("from_numpy(float64):", from_np.dtype)
print("float32 로 맞추기 :", from_np.float().dtype)

# %% CODE 03
# torch.randn(): 평균 0, 표준편차 1 근처의 난수 (원본 표에는 있지만 코드에는 없던 함수)
torch.manual_seed(42)
z = torch.randn(10000)
print("평균      :", round(z.mean().item(), 3))
print("표준편차  :", round(z.std().item(), 3))
