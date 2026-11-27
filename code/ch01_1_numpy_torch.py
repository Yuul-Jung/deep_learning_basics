# ch01_1_numpy_torch.py — 원본: lab_Pytorch/ch01_1_numpy_torch.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
import numpy as np
import torch

print("NumPy version :", np.__version__)
print("PyTorch version:", torch.__version__)

# %% CODE 02
# 같은 데이터를 NumPy 배열과 PyTorch 텐서로 만듭니다.
numbers = [1, 2, 3, 4]

np_array = np.array(numbers, dtype=np.float32)
torch_tensor = torch.tensor(numbers, dtype=torch.float32)

print("NumPy 배열 :", np_array)
print("PyTorch 텐서:", torch_tensor)

print("NumPy 자료형 :", np_array.dtype)
print("PyTorch 자료형:", torch_tensor.dtype)

# %% CODE 03
# 두 라이브러리의 기본 연산 문법은 거의 같습니다.
print("[NumPy]")
print("더하기 :", np_array + 10)
print("곱하기 :", np_array * 2)
print("합계   :", np_array.sum())
print("평균   :", np_array.mean())

print("\n[PyTorch]")
print("더하기 :", torch_tensor + 10)
print("곱하기 :", torch_tensor * 2)
print("합계   :", torch_tensor.sum())
print("평균   :", torch_tensor.mean())

# %% CODE 04
# 2행 3열 데이터
data = [[1, 2, 3],
        [4, 5, 6]]

np_matrix = np.array(data)
torch_matrix = torch.tensor(data)

print("NumPy shape :", np_matrix.shape)
print("PyTorch shape:", torch_matrix.shape)

# 첫 번째 행, 두 번째 열의 값
print("NumPy 값 :", np_matrix[0, 1])
print("PyTorch 값:", torch_matrix[0, 1])

# 2행 3열을 3행 2열로 변경
print("\nNumPy reshape")
print(np_matrix.reshape(3, 2))

print("\nPyTorch reshape")
print(torch_matrix.reshape(3, 2))

# %% CODE 05
a_np = np.array([[1, 2],
                 [3, 4]])

b_np = np.array([[5, 6],
                 [7, 8]])

a_torch = torch.tensor(a_np)
b_torch = torch.tensor(b_np)

# 원소별 곱셈: 같은 위치의 원소끼리 곱합니다.
print("NumPy 원소별 곱셈")
print(a_np * b_np)

print("\nPyTorch 원소별 곱셈")
print(a_torch * b_torch)

# 행렬곱: 행과 열을 곱하여 새로운 행렬을 만듭니다.
print("\nNumPy 행렬곱")
print(a_np @ b_np)

print("\nPyTorch 행렬곱")
print(a_torch @ b_torch)

# %% CODE 06
# NumPy 배열 → PyTorch 텐서
np_data = np.array([10, 20, 30], dtype=np.float32)
tensor_data = torch.tensor(np_data)

# PyTorch 텐서 → NumPy 배열
back_to_numpy = tensor_data.numpy().copy()

print("원본 NumPy :", np_data)
print("PyTorch 변환:", tensor_data)
print("다시 NumPy :", back_to_numpy)

# %% CODE 07
# x에 대한 계산 과정을 추적하도록 설정합니다.
x = torch.tensor(3.0, requires_grad=True)
# y = x² + 2x
y = x**2 + 2*x
# y를 x로 미분합니다.
y.backward()
print("x =", x.item())
print("y =", y.item())
print("dy/dx =", x.grad.item())

# %% CODE 08
# 센서 측정값을 0~1 범위로 변환합니다.
sensor_values = [20.0, 30.0, 40.0, 50.0]

# NumPy
np_sensor = np.array(sensor_values, dtype=np.float32)
np_normalized = (np_sensor - np_sensor.min()) / (np_sensor.max() - np_sensor.min())

# PyTorch
torch_sensor = torch.tensor(sensor_values, dtype=torch.float32)
torch_normalized = (
    (torch_sensor - torch_sensor.min())
    / (torch_sensor.max() - torch_sensor.min())
)

print("NumPy 결과 :", np_normalized)
print("PyTorch 결과:", torch_normalized)

# 두 결과가 같은지 확인합니다.
print("결과 일치:", np.allclose(np_normalized, torch_normalized.numpy()))
