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
| 2 | 선형회귀 | 강의 2장 + 실습 `ch02_1`~`ch02_6` | 작성 예정 |
| 3 | 분류 | 강의 3장 + 실습 `ch03_1`~`ch03_4` | 작성 예정 |
| 4 | 딥러닝: MLP | 강의 4장 + 실습 `ch04_1`~`ch04_6` | 작성 예정 |
| 5 | 합성곱 신경망(CNN) | 강의 5장 + 실습 `ch05_1`~`ch05_2` | 작성 예정 |

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
python3 -m pip install numpy torch mistune latex2mathml
python3 tools/run_blocks.py code/ch01_1_numpy_torch.py code/out/ch01_1_numpy_torch.json   # 파일마다 실행 결과 갱신
python3 tools/make_figures_ch01.py     # 그림
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
