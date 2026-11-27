## 이 부록의 목적과 쓰는 법

1–5장은 문제마다 출력층·손실함수·Optimizer·평가 지표를 정해 놓고 따라 했습니다. 이 부록은 그 선택들을 **한 장의 판단 순서**로 묶습니다. 원본은 「PyTorch 머신러닝·딥러닝 학습 설정 선택 기준」(16쪽) 자료이며, 쪽 순서를 그대로 따릅니다.

- 새 문제를 만났을 때 **무엇부터 정할지** 확인하는 참고 자료입니다.
- 원본에는 코드가 없습니다. 표의 규칙이 실제 PyTorch에서 맞는지 확인하는 짧은 **확인 코드**를 이 부록에서 붙였고, 1–5장의 실행 결과를 예로 연결했습니다.
- 처음 읽을 때는 1–5장을 마친 뒤 복습용으로 보고, 이후에는 문제를 시작할 때마다 마지막 절의 판단 공식만 꺼내 보면 됩니다.

## 핵심 원칙: 모델보다 Target이 먼저

MLP를 쓸지 CNN을 쓸지보다 먼저 확인할 것이 있습니다.

```text
정답 Target 형태 확인 → 문제 유형 판단 → 출력층·Loss 결정 → 학습 조건 조정
```

예를 들어 MNIST를 MLP로 풀든(4장) CNN으로 풀든(5장) 둘 다 **다중분류**입니다. 그래서 둘 다 마지막은 로짓 10개와 `CrossEntropyLoss`입니다.

⟪box:판단 기준⟫

"어떤 모델인가?"가 아니라 "**정답 y가 어떻게 생겼는가?**"가 출력층과 Loss를 정합니다. 모델 이름은 그다음입니다.

⟪/box⟫

## 전체 의사결정 파이프라인

```text
문제 정의 → Target 형태 → 출력층 → Loss → Optimizer → LR·Batch·Epoch → Validation → Metric
```

앞 단계가 뒤 단계를 결정합니다. 특히 앞의 세 칸(Target 형태 → 출력층 → Loss)은 정답 모양이 정하면 거의 자동으로 따라옵니다.

⟪fig:appA_pipeline|그림 A-1 · 학습 설정 결정 순서. 앞 세 칸은 정답 y의 모양이, 가운데 두 칸은 손실 곡선이, 마지막 두 칸은 평가 목적이 정합니다.⟫

- Target이 **실수** → 회귀 → `MSELoss` / `L1Loss`
- Target이 **0/1** → 이진분류 → `BCEWithLogitsLoss`
- Target이 **클래스 번호** → 다중분류 → `CrossEntropyLoss`

## 문제 유형은 Target의 모양으로 구분한다

| 정답 Target | 문제 유형 | Loss |
|---|---|---|
| 실수 | 회귀(Regression) | `MSELoss` / `L1Loss` |
| 0/1 float | 이진분류(Binary) | `BCEWithLogitsLoss` |
| 클래스 번호 long | 다중분류(Multi-class) | `CrossEntropyLoss` |
| 여러 0/1 float | 다중레이블(Multi-label) | `BCEWithLogitsLoss` |

⟪fig:appA_target_map|그림 A-2 · Target 지도. 정답 모양 하나가 출력 수, Loss, 추론 방법, 평가 지표를 함께 정합니다. 오른쪽 끝은 이 교재에서 다룬 장입니다.⟫

모델 구조는 입력 특징을 **처리하는 방식**이고, Loss는 **Target과 출력의 관계**로 결정됩니다. 네 유형의 출력·정답 모양과 Loss를 배치 4개로 확인했습니다(**확인 코드**).

⟪code:appA_extra_checks:01⟫

이진분류와 다중레이블은 같은 `BCEWithLogitsLoss`를 쓰지만 모양이 다릅니다. 이진분류는 `[N, 1]`, 다중레이블은 `[N, C]`입니다. 다중분류만 정답이 **정수(int64 = long)이고 1차원**입니다.

## ① 회귀: 실수값을 예측한다

```text
입력 x (예: 면적 84평, 방 3개) → 회귀 모델 (Linear / MLP) → 예측값 (예: 3억 5천) ↔ 실제값 (예: 3억 6천)
                                                            └ 오차 계산: MSELoss / L1Loss
```

| 평가 Metric | 해석 |
|---|---|
| MAE, RMSE | 작을수록 좋음. 정답과 같은 단위 |
| R² | 1에 가까울수록 좋음. 평균으로 찍는 것보다 못하면 **음수**도 가능 |

이 교재의 예: 2장 실습 2-6 주택가격 — 테스트 MAE 3.500, RMSE 5.629, R² 0.663. `L1Loss`는 오차의 절댓값 평균(= MAE)을 손실로 써서, 큰 오차에 MSE보다 덜 민감합니다.

## ② 이진분류: 두 가지 중 하나

```text
학습: 입력 x → 모델 → 로짓 1개 → BCEWithLogitsLoss
추론: 로짓 → Sigmoid → 확률 → 0.5 기준 → 0 또는 1
```

- **학습 단계에서는 Sigmoid를 모델에 넣지 않습니다.** `BCEWithLogitsLoss`가 안에서 계산합니다.
- Target은 `[N, 1]` float, 값은 `0.0` / `1.0`을 씁니다.

이 교재의 예: 3장 실습 3-1·3-2. 극단적인 로짓에서 `Sigmoid → BCELoss`는 손실을 100에서 자르고 `BCEWithLogitsLoss`는 정확한 값을 낸다는 것을 3장 추가 확인에서 보았습니다. 0.5는 기본값일 뿐이고, 질환 선별처럼 놓치는 비용이 큰 문제는 임계값을 낮춥니다(실습 3-2의 임계값 비교).

## ③ 다중분류: 여러 클래스 중 하나

```text
입력 이미지/특징 → 모델 → 이미지마다 로짓 10개 (전체 출력 [4, 10]) → CrossEntropyLoss
예: output [4, 10], target tensor([7, 2, 1, 9])
```

입력은 로짓 `[N, C]`, Target은 원핫이 아니라 **클래스 번호 `[N]` long**을 씁니다.

⟪code:appA_extra_checks:03⟫

참고로 지금의 PyTorch는 원핫 float `[N, C]`를 넣어도 "클래스별 확률"로 해석해 같은 값을 냅니다. 그래도 정답 클래스가 하나인 문제에서는 번호 `[N]`가 표준이고 메모리도 적게 씁니다. 이 교재의 예: 3장 실습 3-3·3-4, 4장 MNIST·Fashion-MNIST, 5장 CNN.

## ④ 다중레이블: 여러 정답이 동시에 가능

사진 한 장에 사람·자동차·신호등이 함께 있을 수 있습니다.

| | 사람 | 자동차 | 강아지 | 신호등 |
|---|---|---|---|---|
| 사진 1장 | 1 | 1 | 0 | 1 |

```text
입력 → 모델 → 로짓 C개 → 클래스마다 독립 Sigmoid → BCEWithLogitsLoss
```

정답이 하나만 있는 것이 아닙니다. Target은 `[N, C]` float, 각 값은 0/1이며 **각 라벨이 독립적으로** 동작합니다. 그래서 소프트맥스가 아니라 칸마다 시그모이드를 씁니다(3장의 "시그모이드와 소프트맥스" 표).

⟪code:appA_extra_checks:04⟫

확률의 합이 2.53입니다. 각 칸이 "이 라벨이 있는가"를 따로 판단하므로 합이 1일 필요가 없습니다. 1–5장에는 다중레이블 실습이 없으며, 이 표는 문제 유형을 구분하기 위한 참고입니다.

## Loss Function 선택 요약

| 문제 유형 | Loss |
|---|---|
| 회귀 | `MSELoss` / `L1Loss` |
| 이진분류 | `BCEWithLogitsLoss` |
| 다중분류 | `CrossEntropyLoss` |
| 다중레이블 | `BCEWithLogitsLoss` |

### PyTorch에서 특히 주의할 조합

| Loss | 입력 | 안에 포함된 계산 |
|---|---|---|
| `BCELoss` | Sigmoid **후** 확률값 | 없음 |
| `BCEWithLogitsLoss` | **로짓** | Sigmoid |
| `CrossEntropyLoss` | **로짓** | (Log)Softmax |

정답 모양이나 입력 범위를 잘못 맞추면 PyTorch가 바로 오류를 냅니다(**확인 코드**).

⟪code:appA_extra_checks:02⟫

⟪box:실무 기본⟫

모델 마지막은 **로짓을 그대로** 내보내고, Loss 안의 Sigmoid/Softmax 계산을 활용합니다. 그리고 **Loss ≠ Metric**입니다. Loss는 학습을 위한 값, Metric은 사람이 성능을 판단하기 위한 값입니다.

⟪/box⟫

## Optimizer는 "수정 방법"을 결정한다

Loss는 **얼마나 틀렸는가**, Optimizer는 **W와 b를 어떻게 고칠 것인가**를 담당합니다.

| Optimizer | 한 줄 특징 | 쓰는 때 |
|---|---|---|
| SGD | 기울기 방향으로 이동, 기본 원리 | 학습 원리 이해 (2·3장) |
| SGD + Momentum | 이전 이동 방향을 관성처럼 반영, 진동 감소 | 진동 안정화 |
| Adam | 매개변수마다 이동 크기를 조절 | 초기 실험·빠른 실험 (4·5장) |
| AdamW | Weight Decay(가중치를 작게 유지하는 규제)를 Adam의 갱신과 **분리**해 적용 | 대규모 모델, Transformer 계열에서 자주 사용 |

⟪code:appA_extra_checks:06⟫

네 Optimizer 모두 같은 `model.parameters()`를 받습니다. 바뀌는 것은 **고치는 규칙**뿐입니다. 이 교재의 예: 4장 실습 4-2에서 같은 조건으로 비교했을 때, 이번 실행에서는 SGD + Momentum(최고 검증 정확도 0.9732)이 Adam(0.9692)보다 높았습니다. Optimizer의 우열은 문제와 설정에 따라 바뀌므로 검증 결과로 고릅니다.

## Learning Rate · Batch Size · Epoch

| 설정 | 너무 작으면 | 너무 크면 | 이 교재의 예 |
|---|---|---|---|
| Learning Rate | 느림 | 진동·폭발 | 2장 실습 2-3: α 0.01은 10번 뒤 W 2.50(느림), 0.3은 1,429(발산) |
| Batch Size | 기울기 변동 ↑ | 기울기 안정 ↑, 메모리 ↑ | 4장: 배치 64 → Epoch당 갱신 860번 |
| Epoch | 덜 배움 | 무조건 늘리면 과적합 가능(Val Loss 상승) | 4장 실습 4-3: 기본 MLP는 7 Epoch 뒤 검증 손실이 더 내려가지 않음 |

⟪box:조정 원칙⟫

반드시 **Loss 그래프와 Validation 결과를 함께** 보면서 조정합니다. 한 번에 하나씩 바꾸고, 바꾼 이유와 결과를 기록합니다.

⟪/box⟫

## 데이터 전처리: Train 기준만 사용한다

| | 과정 | 결과 |
|---|---|---|
| 정상 | Train 데이터에서 mean·std 계산 → Train / Val / Test에 **같은 기준**으로 변환 | 공정한 평가 |
| 잘못 | Test 데이터에서 Test mean·std를 따로 계산 | **데이터 누출**(Data Leakage) |

Test는 마지막까지 "처음 보는 데이터"처럼 유지합니다. 이 교재의 예: 2장 그림 2-6과 실습 2-4·2-6, 3장 실습 3-2·3-4의 표준화, 4장 실습 4-4·4-6의 Checkpoint에 정규화 평균·표준편차를 함께 저장한 것.

## 학습·검증·테스트 데이터 분리

| 데이터 | 하는 일 |
|---|---|
| 실제 학습용 Train | 가중치 학습 |
| 검증용 Validation | 학습 중 모델 점검, 과적합 확인, 하이퍼파라미터 조정 |
| 테스트 Test | 최종 성능 평가에만 사용 |

원본의 예: 전체 100% → Train 80%, Test 20% / Train 80% 안에서 다시 Train 64%, Validation 16%. 데이터 70,000개라면 이렇게 됩니다(**확인 코드**).

⟪code:appA_extra_checks:07⟫

| 상태 | Train Loss | Validation Loss |
|---|---|---|
| 정상 학습 | ↓ | ↓ |
| Underfitting(과소적합) | 높음 | 높음 |
| Overfitting(과적합) | ↓ | ↑ |

이 교재의 예: 4장 MNIST는 원래 훈련 60,000장에서 검증 5,000장을 떼어 냈습니다. 비율은 데이터 크기에 따라 정하며, 검증 데이터가 수천 개 이상이면 비율이 작아도 충분히 안정적입니다.

## MNIST: MLP와 CNN은 앞부분만 다르다

같은 다중분류 문제라면 마지막 구조는 같습니다.

```text
MLP : 28×28 → Flatten → Linear/ReLU            ┐
                                                ├→ 10개 중 하나 선택 → 로짓 10개 → CrossEntropyLoss
CNN : 28×28 → Conv/ReLU/Pooling → Flatten       ┘
      입력 처리 방식: 다름                        출력층과 Loss: 같음
```

이 교재의 예: 4장 실습 4-5(MLP)와 5장 실습 5-1(CNN)은 같은 Fashion-MNIST에 같은 Loss·Optimizer·Epoch를 썼습니다. 다른 것은 앞부분뿐이었고, 그 결과 정확도는 비슷(0.8848 대 0.8833)했지만 파라미터 수(235,146 대 9,098)와 2픽셀 이동에 대한 강건함(정확도 감소 33%p 대 14%p)이 크게 달랐습니다.

## Metric은 문제 유형에 맞춰 해석한다

| 정답 | 문제 유형 | Metric |
|---|---|---|
| 실수 예측 | 회귀 | MAE / RMSE / R² |
| 0/1 분류 | 이진분류 | Accuracy / Precision / Recall / F1 |
| 클래스 번호 | 다중분류 | Accuracy / Macro F1 |
| 여러 0/1 라벨 | 다중레이블 | Micro F1 / Macro F1 |

- **Macro F1** — 클래스마다 F1을 구해 단순 평균. 적은 클래스도 같은 무게로 반영됩니다.
- **Micro F1** — 모든 클래스의 TP·FP·FN을 합쳐 한 번에 계산. 많은 클래스의 영향이 큽니다.

Loss는 학습용이고 Metric은 평가용입니다. **불균형 데이터는 Accuracy만 보면 위험합니다.** 100개 중 양성이 5개인데 모델이 전부 0이라고 답하는 경우를 계산했습니다(**확인 코드**).

⟪code:appA_extra_checks:05⟫

정확도는 0.95로 좋아 보이지만 양성을 하나도 찾지 못했습니다(클래스 1 F1 0). Macro F1 0.487이 이 문제를 드러냅니다. 3장 실습 3-2에서 정확도와 함께 재현율·F1을 본 이유이고, 4·5장에서 클래스별 정확도를 따로 본 이유입니다.

## 핵심 판단 공식

새로운 문제를 만나면 **Loss 이름부터 떠올리지 않습니다.**

| y가 | 문제 유형 | Loss |
|---|---|---|
| 실수 | Regression | `MSELoss` / `L1Loss` |
| 0/1 float | Binary | `BCEWithLogitsLoss` |
| 클래스 번호 long | Multi-class | `CrossEntropyLoss` |
| 여러 0/1 float | Multi-label | `BCEWithLogitsLoss` |

```text
Target 형태 → 문제 유형 → 출력 노드 수 → Logit 형태 → Loss → Optimizer → Validation → Metric
```

⟪box:스스로 점검⟫

1. 내일 기온(℃)을 예측한다. 출력 노드 수와 Loss는?
2. 메일이 스팸인지 판별한다. 출력 노드 수, 정답 모양, Loss는?
3. 사진 속 동물이 고양이·개·토끼 중 하나다. 정답 모양과 Loss는?
4. 영화에 붙는 장르(액션·코미디·로맨스·SF)를 모두 고른다. 정답 모양과 Loss, 추론 방법은?
5. 모델 마지막에 `nn.Softmax()`를 넣고 `CrossEntropyLoss`를 쓰면 무엇이 문제인가?
6. 양성 1%인 데이터에서 정확도 99%가 나왔다. 무엇을 더 확인해야 하는가?
7. 테스트 데이터의 평균으로 테스트 데이터를 표준화하면 안 되는 이유는?

정답: ① 1개, `MSELoss`(또는 `L1Loss`) ② 1개, `[N, 1]` float 0/1, `BCEWithLogitsLoss` ③ `[N]` long 클래스 번호, `CrossEntropyLoss` ④ `[N, 4]` float 0/1, `BCEWithLogitsLoss`, 칸마다 시그모이드 후 0.5 기준 ⑤ 소프트맥스가 두 번 적용되어 학습이 잘못됨, 모델은 로짓을 내보내야 함 ⑥ 양성 클래스의 재현율·정밀도·F1(Macro F1), 혼동행렬 ⑦ 평가해야 할 정보가 전처리에 섞이는 데이터 누출

⟪/box⟫

### 원본 자료와 달라진 점

| 원본 | 이 부록 |
|---|---|
| 슬라이드 형식(16쪽)이며 코드가 없음 | 쪽 순서대로 글과 표로 옮기고, 규칙을 PyTorch로 확인하는 확인 코드 7개를 붙임 |
| 각 쪽의 그림(파이프라인, Target 지도, 흐름도) | 직접 만든 그림 2개(그림 A-1, A-2)와 표·흐름 글로 대신함 |
| 각 규칙에 실습 예가 없음 | 1–5장의 실제 실행 결과를 예로 연결함 |
| 다중분류 Target은 "원핫이 아니라 클래스 번호"라고만 설명 | 원핫 float도 확률로 해석되어 같은 값이 나온다는 점을 확인하고, 그래도 번호가 표준인 이유를 적음 |
| Micro F1 / Macro F1, AdamW의 Weight Decay를 용어만 제시 | 뜻을 한 줄씩 풀어 쓰고, Macro F1은 불균형 예제로 계산함 |
