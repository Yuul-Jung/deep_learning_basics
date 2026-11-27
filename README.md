# 딥러닝 기초 — NumPy·텐서에서 CNN까지

NumPy와 PyTorch 텐서에서 출발해 선형회귀, 분류, 다층 퍼셉트론(MLP), 합성곱 신경망(CNN)까지 강의와 실습을 한 장씩 이어 가는 학습 안내서입니다. 나스(engworks) AI 테크트리의 2단계 과정입니다.

- 첫 화면: [`dist/index.html`](dist/index.html)

## 📘 교육용 HTML 공통 제작 규칙

이 저장소의 교육자료(웹페이지·강의 HTML)는 [Yuul-Jung 교육용 HTML 공통 표준](https://github.com/Yuul-Jung/rule_md/tree/main/web/education-html)을 적용합니다.

- [제작 규칙](https://github.com/Yuul-Jung/rule_md/blob/main/web/education-html/STANDARD.md)
- [다른 저장소 적용 안내](https://github.com/Yuul-Jung/rule_md/blob/main/web/education-html/INTEGRATION.md)
- [기준 디자인 사례](https://github.com/Yuul-Jung/ai-agent-engineering)

공통 책장 이미지, 컴팩트한 상단 배너, 좌측 전체 장/절 탐색, 모바일 메뉴, 파일 버전 및 검증 기준을 따릅니다.

## 장 구성

| 장 | 제목 | 원본 자료 | 상태 |
|---|---|---|---|
| 1 | NumPy와 PyTorch 텐서 | 실습 `ch01_1`~`ch01_3` | v01 |
| 2 | 선형회귀 | 강의 2장 + 실습 `ch02_1`~`ch02_6` | v01 |
| 3 | 분류 | 강의 3장 + 실습 `ch03_1`~`ch03_4` | v01 |
| 4 | 딥러닝: MLP | 강의 4장 + 실습 `ch04_1`~`ch04_6` | v01 |
| 5 | 합성곱 신경망(CNN) | 강의 5장 + 실습 `ch05_1`~`ch05_2` | v01 |

## 폴더

| 폴더·파일 | 내용 |
|---|---|
| `content_src/chNN.md` | 장 원고 (여기를 고침) |
| `content/chNN.md` | 코드·실행 결과를 펼친 원고 (생성물) |
| `code/chNN_K_*.py` | 원본 실습의 CODE 블록을 순서대로 옮긴 코드 |
| `code/chNN_extra_checks.py` | 원본 밖의 추가 확인 코드 |
| `code/out/*.json` | 블록별 실제 실행 결과와 실행 환경(`env.json`) |
| `figures/` | 본문 그림 SVG (생성물, 원본은 `tools/make_figures_chNN.py`) |
| `tools/` | 실행·펼치기·그림 생성 도구 |
| `dist/` | 배포용 HTML (첫 화면 + 장별 `_vNN.html`) |

## 다시 만들기

```bash
python3 -m pip install numpy torch torchvision matplotlib pandas scikit-learn mistune latex2mathml
# 4장부터 MNIST·Fashion-MNIST를 code/data/ 에 내려받고, 학습한 모델은 code/models/ 에 저장한다(둘 다 git 제외)
python3 tools/run_blocks.py code/ch01_1_numpy_torch.py   # 파일마다 실행 결과 갱신 → code/out/<파일>.json
python3 tools/run_blocks.py code/ch02_4_linear_regression.py              # 그림이 있는 블록은 figures/에 SVG도 저장
(cd tools && for f in make_figures_ch0*.py; do python3 $f; done)   # 설명 그림
python3 tools/expand_code.py           # content_src → content
python3 build.py                       # dist/ 생성
```

## 검증 (1장 v01)

| 항목 | 판정 | 내용 |
|---|---|---|
| 코드 실행 | 통과 | 실습 26블록 + 추가 확인 3블록을 Python 3.13 · NumPy 2.5 · PyTorch 2.15(CPU)에서 실행, 본문 결과는 실행값 그대로 |
| 원본 대조 | 통과 | 실습 PDF 3개의 CODE 블록과 순서 일치 |
| 화면 | 통과 | 1366 · 390px에서 가로 넘침 0, 스크립트 오류 0, 좌측 목차·절 목차 생성 |
| GPU 실행 | 미검증 | 실행 환경에 GPU 없음. `device` 결과는 `cpu` |
| GitHub Pages 실제 배포 | 미검증 | Pages 설정 후 확인 필요 |

## 검증 (2장 v01)

| 항목 | 판정 | 내용 |
|---|---|---|
| 코드 실행 | 통과 | 실습 63블록 + 추가 확인 3블록 실행, 그래프 10개는 실행 중 저장한 SVG를 그대로 실음 |
| 원본 대조 | 통과 | 실습 PDF 6개의 CODE 블록과 순서 일치, 강의 45장 순서대로 배치 |
| 수치 대조 | 통과 | 본문의 손 계산(MSE 1.1667, 기울기 37.33, 학습률 경계 0.214 등)을 실행 결과와 대조 |
| 화면 | 통과 | 1366 · 390px에서 가로 넘침 0, 스크립트 오류 0 |
| GitHub Pages 실제 배포 | 미검증 | Pages 설정 후 확인 필요 |

## 검증 (3장 v01)

| 항목 | 판정 | 내용 |
|---|---|---|
| 코드 실행 | 통과 | 실습 53블록 + 추가 확인 4블록 실행(pandas 3.0 · scikit-learn 1.9 포함), 그래프 12개는 실행 중 저장한 SVG |
| 원본 대조 | 통과 | 실습 PDF 4개의 CODE 블록과 순서 일치(3-2 CODE 12는 쪽 경계로 끊겨 공백까지 같은 노트북 셀로 확인), 강의 33장 순서대로 배치 |
| 수치 대조 | 통과 | 시그모이드·BCE·소프트맥스·교차엔트로피·혼동행렬 지표를 실행 결과와 대조 |
| 화면 | 통과 | 1366 · 390px에서 가로 넘침 0, 스크립트 오류 0 |
| 1·2장 재실행 | 통과 | 3장 작업 전 1·2장 전체 코드를 다시 실행해 실행 결과·그림이 바이트 단위로 같음을 확인 |

## 검증 (4장 v01)

| 항목 | 판정 | 내용 |
|---|---|---|
| 코드 실행 | 통과 | 실습 123블록(4-1 16, 4-2 14, 4-3 25, 4-4 24, 4-5 30, 4-6 14) + 추가 확인 4블록을 CPU에서 실행(torchvision 0.29), 그래프는 실행 중 저장한 SVG |
| 원본 대조 | 통과(보완 표시) | 실습 PDF 4개·노트북 2개와 순서 일치. PDF 쪽 아래로 잘린 3블록은 원본 노트북·PDF 내부 글자로 되살리거나 `(보완)` 주석으로 채움(본문 "원본 자료와 달라진 점") |
| 수치 대조 | 통과 | 파라미터 수(101,770 · 535,818 · 235,146), Iteration 860 · 4,300, 혼동행렬 지표를 실행 결과와 대조 |
| 화면 | 통과 | 1366 · 390px에서 가로 넘침 0, 스크립트 오류 0 |
| GPU 실행 | 미검증 | 실행 환경에 GPU 없음 |

## 검증 (5장 v01)

| 항목 | 판정 | 내용 |
|---|---|---|
| 코드 실행 | 통과 | 실습 53블록(5-1 33, 5-2 20) + 추가 확인 7블록을 CPU에서 실행, 그래프는 실행 중 저장한 SVG |
| 원본 대조 | 통과 | 실습 PDF 2개의 CODE 블록과 순서 일치(노트북과 다른 3곳은 PDF를 따름), 강의 25장 순서대로 배치 |
| 수치 대조 | 통과 | 강의 슬라이드의 합성곱·풀링·패딩·출력 크기 숫자를 F.conv2d로 재계산해 모두 일치, 파라미터 9,098 확인 |
| 화면 | 통과 | 1366 · 390px에서 가로 넘침 0, 스크립트 오류 0 |
| GPU 실행 | 미검증 | 실행 환경에 GPU 없음 |
