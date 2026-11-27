# appA_extra_checks.py — 부록 A(학습 설정 선택 기준) 확인 코드
# 원본 자료(PDF 16쪽)에는 코드가 없으므로, 표의 규칙을 PyTorch로 직접 확인하는 코드만 둔다.

# %% CODE 01
# 네 가지 문제 유형의 출력·정답 모양과 Loss 조합 (배치 N=4)
import torch
from torch import nn

torch.manual_seed(0)
N = 4

cases = {
    "회귀(Regression)": (torch.randn(N, 1), torch.tensor([[3.5], [2.1], [4.0], [1.2]]), nn.MSELoss()),
    "이진분류(Binary)": (torch.randn(N, 1), torch.tensor([[1.0], [0.0], [1.0], [0.0]]), nn.BCEWithLogitsLoss()),
    "다중분류(Multi-class)": (torch.randn(N, 10), torch.tensor([7, 2, 1, 9]), nn.CrossEntropyLoss()),
    "다중레이블(Multi-label)": (torch.randn(N, 4), torch.tensor([[1.0, 1.0, 0.0, 1.0],
                                                             [0.0, 1.0, 0.0, 0.0],
                                                             [1.0, 0.0, 1.0, 0.0],
                                                             [0.0, 0.0, 0.0, 1.0]]), nn.BCEWithLogitsLoss()),
}
for name, (output, target, loss_function) in cases.items():
    loss = loss_function(output, target)
    print(f"{name:<20} 출력 {str(list(output.shape)):<8} 정답 {str(list(target.shape)):<8} "
          f"{str(target.dtype):<14} {type(loss_function).__name__:<18} Loss {loss.item():.4f}")

# %% CODE 02
# 자주 하는 실수: 정답 모양·자료형이 Loss와 맞지 않으면 PyTorch가 오류를 낸다
mistakes = [
    ("CrossEntropyLoss에 [N, 1] 정답", nn.CrossEntropyLoss(), torch.randn(N, 10), torch.tensor([[7], [2], [1], [9]])),
    ("BCEWithLogitsLoss에 long 정답", nn.BCEWithLogitsLoss(), torch.randn(N, 1), torch.tensor([[1], [0], [1], [0]])),
    ("BCELoss에 로짓(범위 밖 값)", nn.BCELoss(), torch.tensor([[2.0], [-1.0], [0.5], [3.0]]), torch.tensor([[1.0], [0.0], [1.0], [0.0]])),
]
for name, loss_function, output, target in mistakes:
    try:
        loss_function(output, target)
        print(f"{name:<32} → 오류 없음")
    except (RuntimeError, ValueError) as error:
        print(f"{name:<32} → {type(error).__name__}: {str(error).splitlines()[0][:70]}")

# %% CODE 03
# CrossEntropyLoss의 정답은 클래스 번호 [N] long이 기본이다.
# (원핫 float [N, C]를 넣으면 '클래스별 확률'로 해석되어 같은 값이 나오지만, 교재에서는 번호를 쓴다)
logits = torch.randn(N, 10)
labels = torch.tensor([7, 2, 1, 9])
one_hot = nn.functional.one_hot(labels, num_classes=10).float()
print("정답 번호 [N] long  :", round(nn.CrossEntropyLoss()(logits, labels).item(), 6))
print("원핫 [N, 10] float :", round(nn.CrossEntropyLoss()(logits, one_hot).item(), 6))

# %% CODE 04
# 다중레이블: 클래스마다 독립 시그모이드 → 확률의 합이 1이 아니어도 된다
label_names = ["사람", "자동차", "강아지", "신호등"]
photo_logits = torch.tensor([[2.1, 1.3, -1.8, 0.9]])
probabilities = torch.sigmoid(photo_logits)[0]
predicted = (probabilities >= 0.5).int()
for name, p, y in zip(label_names, probabilities.tolist(), predicted.tolist()):
    print(f"{name:<4} 확률 {p:.3f} → {y}")
print("확률 합:", round(probabilities.sum().item(), 3), "(소프트맥스였다면 1)")

# %% CODE 05
# 불균형 데이터에서 Accuracy만 보면 위험하다: 100개 중 양성 5개, 모델이 전부 0이라고 답하는 경우
from collections import Counter


def f1_for(cls, y_true, y_pred):
    tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
    fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
    fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


y_true = [0] * 95 + [1] * 5
y_pred = [0] * 100
accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
f1_0, f1_1 = f1_for(0, y_true, y_pred), f1_for(1, y_true, y_pred)
print("정답 분포:", dict(Counter(y_true)))
print(f"Accuracy {accuracy:.2f} | 클래스 0 F1 {f1_0:.3f} | 클래스 1 F1 {f1_1:.3f} | Macro F1 {(f1_0 + f1_1) / 2:.3f}")

# %% CODE 06
# Optimizer는 같은 모델 파라미터를 받고 '고치는 방법'만 다르다 (만들기만 하고 학습하지 않음)
model = nn.Linear(784, 10)
optimizers = {
    "SGD": torch.optim.SGD(model.parameters(), lr=0.1),
    "SGD + Momentum": torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
    "Adam": torch.optim.Adam(model.parameters(), lr=0.001),
    "AdamW": torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01),
}
for name, optimizer in optimizers.items():
    group = optimizer.param_groups[0]
    print(f"{name:<15} lr={group['lr']:<6} momentum={group.get('momentum', '-')!s:<4} weight_decay={group['weight_decay']}")

# %% CODE 07
# 전체 100% → Train 80% / Test 20%, Train 80% 안에서 다시 Train 64% / Validation 16%
total = 70_000
train_all, test = int(total * 0.8), total - int(total * 0.8)
train, validation = int(train_all * 0.8), train_all - int(train_all * 0.8)
print(f"전체 {total:,} → Train {train_all:,} + Test {test:,}")
print(f"Train {train_all:,} → 실제 학습 {train:,} ({train / total:.0%}) + Validation {validation:,} ({validation / total:.0%})")
