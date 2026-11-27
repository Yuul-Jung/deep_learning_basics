# ch01_3_tensor_operations.py — 원본: lab_Pytorch/ch01_3_tensor_operations.pdf 의 CODE 블록을 순서대로 옮김

# %% CODE 01
import numpy as np
import torch

# %% CODE 02
# 같은 난수 결과를 얻기 위한 기준값
torch.manual_seed(42)
zeros_tensor = torch.zeros(2, 3)
ones_tensor = torch.ones(2, 3)
range_tensor = torch.arange(0, 6).reshape(2, 3)
random_tensor = torch.rand(2, 3)
print("zeros:")
print(zeros_tensor)
print()
print("ones:")
print(ones_tensor)
print()
print("arange + reshape:")
print(range_tensor)
print()
print("rand:")
print(random_tensor)

# %% CODE 03
numpy_data = np.array([[10, 20], [30, 40]], dtype=np.float32)
shared_tensor = torch.from_numpy(numpy_data)
copied_tensor = shared_tensor.clone()
# NumPy 배열의 첫 번째 값을 변경합니다.
numpy_data[0, 0] = 99
print("변경된 NumPy 배열:")
print(numpy_data)
print()
print("메모리를 공유하는 Tensor:")
print(shared_tensor)
print()
print("미리 복사한 독립 Tensor:")
print(copied_tensor)

# %% CODE 04
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
data_on_device = data.to(device)
print("선택된 장치:", device)
print("Tensor의 장치:", data_on_device.device)

# %% CODE 05
tensor_a = torch.arange(1, 7).reshape(2, 3)
print("원래 Tensor:")
print(tensor_a)
print("shape:", tensor_a.shape)
print()
print("reshape(3, 2):")
print(tensor_a.reshape(3, 2))
print()
print("flatten():")
print(tensor_a.flatten())

# %% CODE 06
tensor_b = torch.tensor(
    [[10, 20, 30],
     [40, 50, 60]]
)
cat_dim0 = torch.cat([tensor_a, tensor_b], dim=0)
cat_dim1 = torch.cat([tensor_a, tensor_b], dim=1)
print("cat dim=0:")
print(cat_dim0)
print("shape:", cat_dim0.shape)
print()
print("cat dim=1:")
print(cat_dim1)
print("shape:", cat_dim1.shape)

# %% CODE 07
stack_dim0 = torch.stack([tensor_a, tensor_b], dim=0)
print("stack dim=0:")
print(stack_dim0)
print("shape:", stack_dim0.shape)

# %% CODE 08
# 첫 번째 측정 묶음
batch_1 = torch.tensor(
    [[22.0, 0.10],
     [25.0, 0.15]]
)
# 두 번째 측정 묶음
batch_2 = torch.tensor(
    [[31.0, 0.45],
     [36.0, 0.90]]
)
# 모든 측정값을 하나의 표로 연결합니다.
all_sensor_data = torch.cat([batch_1, batch_2], dim=0)
# 두 측정 묶음을 구분한 상태로 쌓습니다.
grouped_sensor_data = torch.stack([batch_1, batch_2], dim=0)
print("하나로 연결한 센서 데이터:")
print(all_sensor_data)
print("shape:", all_sensor_data.shape)
print()
print("측정 묶음별로 쌓은 센서 데이터:")
print(grouped_sensor_data)
print("shape:", grouped_sensor_data.shape)
