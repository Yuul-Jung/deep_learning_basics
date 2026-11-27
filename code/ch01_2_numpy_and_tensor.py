# ch01_2_numpy_and_tensor.py — 원본: lab_Pytorch/ch01_2_numpy_and_tensor.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
import numpy as np
import torch

# %% CODE 02
# Python 리스트
list_a = [1, 2, 3]
list_b = [10, 20, 30]

# NumPy 배열
array_a = np.array([1, 2, 3])
array_b = np.array([10, 20, 30])

print("Python 리스트 + :", list_a + list_b)
print("NumPy 배열 +    :", array_a + array_b)

# %% CODE 03
scalar = np.array(7)
vector = np.array([1, 2, 3])
matrix = np.array([[1, 2], [3, 4]])

print("Scalar → ndim:", scalar.ndim, ", shape:", scalar.shape)
print("Vector → ndim:", vector.ndim, ", shape:", vector.shape)
print("Matrix → ndim:", matrix.ndim, ", shape:", matrix.shape)

# %% CODE 04
sensor_data = np.array(
    [
        [22.5, 0.10],
        [31.0, 0.45],
        [36.0, 0.90],
    ],
    dtype=np.float32,
)

print(sensor_data)
print("차원 수:", sensor_data.ndim)
print("shape:", sensor_data.shape)
print("자료형:", sensor_data.dtype)

# %% CODE 05
print("첫 번째 측정 데이터:", sensor_data[0])
print("모든 온도값:", sensor_data[:, 0])
print("모든 진동값:", sensor_data[:, 1])
print("두 번째 온도값:", sensor_data[1, 0])

# %% CODE 06
temperatures = sensor_data[:, 0]

high_mask = temperatures >= 30.0
high_temperatures = temperatures[high_mask]

print("30℃ 이상 여부:", high_mask)
print("30℃ 이상 온도:", high_temperatures)

# %% CODE 07
temperature_offset = 1.5
corrected_temperatures = temperatures + temperature_offset

print("보정 전:", temperatures)
print("보정 후:", corrected_temperatures)

# %% CODE 08
sensor_tensor = torch.from_numpy(sensor_data)

print(sensor_tensor)
print("자료형:", type(sensor_tensor))
print("dim:", sensor_tensor.dim())
print("shape:", sensor_tensor.shape)
print("dtype:", sensor_tensor.dtype)

# %% CODE 09
print("NumPy ndim :", sensor_data.ndim)
print("Tensor dim :", sensor_tensor.dim())

print("NumPy shape:", sensor_data.shape)
print("Tensor shape:", sensor_tensor.shape)

print("Tensor device:", sensor_tensor.device)

# %% CODE 10
practice_data = np.array(
    [
        [24.0, 0.20],
        [34.0, 0.55],
        [38.0, 0.85],
    ],
    dtype=np.float32,
)

practice_tensor = torch.from_numpy(practice_data)

temperature_alarm = practice_tensor[:, 0] >= 35.0
vibration_alarm = practice_tensor[:, 1] >= 0.7
inspection_required = temperature_alarm & vibration_alarm

print("센서 데이터:")
print(practice_tensor)

print()
print("점검 필요 여부:")
print(inspection_required)

print()
print("점검이 필요한 데이터:")
print(practice_tensor[inspection_required])
