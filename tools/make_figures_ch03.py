"""3장 그림(SVG) 생성기 — figures/ch03_*.svg

곡선 좌표와 확률·손실 값은 이 스크립트가 식으로 계산한다(손으로 찍은 점 없음).
"""
import math
from svgkit import *  # noqa: F401,F403

RED, ORG = '#B3261E', '#C77700'


# ── 그림 3-1. 3장 큰 그림 ───────────────────────────────────────
def fig_overview():
    W, H = 760, 380
    b = [text(380, 28, '3장의 흐름 — 2장의 직선 뒤에 "확률로 바꾸는 함수"를 붙인다', 16, 600)]
    cols = [('회귀 (2장)', 'z = XW + b', '그대로 출력', 'MSE', '숫자 하나', INK),
            ('이진분류', 'z = XW + b', 'sigmoid(z)', 'BCE', '0 또는 1', PUR),
            ('다중분류', 'Z = XWᵀ + b', 'softmax(Z)', '교차엔트로피', 'K개 중 하나', TEAL)]
    for i, (t, f, a, l, o, c) in enumerate(cols):
        x = 30 + i * 245
        b.append(box(x, 50, 225, 172, fill='#FFFFFF', stroke=c, sw=2))
        b.append(text(x + 112, 76, t, 15, 700, c))
        rows = [('가설', f), ('출력 변환', a), ('손실', l), ('답', o)]
        for k, (h, v) in enumerate(rows):
            y = 98 + k * 30
            b.append(text(x + 16, y + 4, h, 12, 500, '#4A4F5A', 'start'))
            b.append(text(x + 210, y + 4, v, 12.5, 600, INK, 'end', mono=(k < 2)))
            if k < 3:
                b.append(f'<line x1="{x + 12}" y1="{y + 15}" x2="{x + 213}" y2="{y + 15}" stroke="#E6E7E2"/>')
        if i < 2:
            b.append(arrow(x + 227, 136, x + 243, 136))
    b.append(text(380, 248, '학습 반복 다섯 줄(순전파 → 손실 → zero_grad → backward → step)은 그대로', 12.5, 500, RED))
    b.append(text(380, 280, '강의와 실습의 대응', 13.5, 600))
    rows = [('강의 3-1 이진분류 · 로지스틱 가설 · 새 손실함수', '실습 3-1 로지스틱 분류 · 3-2 심장질환'),
            ('강의 3-2 다중분류 · 소프트맥스 · 교차엔트로피', '실습 3-3 Softmax 다중분류 · 3-4 Iris')]
    for i, (l, r) in enumerate(rows):
        y = 296 + i * 36
        b.append(box(20, y, 360, 28, fill=SOFT, stroke=PUR, sw=1, rx=5)); b.append(text(200, y + 19, l, 12.5, 600, PUR))
        b.append(arrow(382, y + 14, 418, y + 14))
        b.append(box(422, y, 318, 28, fill=TSOFT, stroke=TEAL, sw=1, rx=5)); b.append(text(581, y + 19, r, 12.5, 500, INK))
    svg('ch03_overview', W, H, '3장 큰 그림',
        '회귀, 이진분류, 다중분류의 가설·출력 변환·손실·답을 나란히 비교하고 강의 두 부분과 실습 네 개의 대응을 보여 준다', ''.join(b))


# ── 그림 3-2. 시그모이드 ─────────────────────────────────────────
def fig_sigmoid():
    W, H = 760, 300
    b = [text(380, 26, '시그모이드 σ(z) = 1 / (1 + e⁻ᶻ) — 어떤 로짓이든 0과 1 사이로 접는다', 15, 600)]
    fr, X, Y = plot_frame(70, 50, 380, 190, (-8, 8), (0, 1), [-8, -4, 0, 4, 8], [0, 0.5, 1], '로짓 z', '확률 σ(z)')
    b.append(fr)
    b.append(poly([(X(-8 + i * 0.1), Y(1 / (1 + math.exp(-(-8 + i * 0.1))))) for i in range(161)], PUR, 2.4))
    b.append(f'<line x1="{X(-8)}" y1="{Y(0.5)}" x2="{X(8)}" y2="{Y(0.5)}" stroke="{RED}" stroke-dasharray="5 4"/>')
    for z in (-5, 0, 5):
        p = 1 / (1 + math.exp(-z))
        b.append(dot(X(z), Y(p), INK, 4.5))
    # 표
    b.append(text(600, 70, '실습 3-1 CODE 02의 세 점', 13, 600))
    rows = [('로짓 z', '확률', '클래스')] + [(f'{z:g}', f'{1 / (1 + math.exp(-z)):.4f}', '1' if 1 / (1 + math.exp(-z)) >= 0.5 else '0') for z in (-5, 0, 5)]
    for r, row in enumerate(rows):
        y = 96 + r * 30
        for c, v in enumerate(row):
            x = 500 + c * 72
            b.append(box(x, y - 20, 72, 30, fill=GRAY if r == 0 else '#FFFFFF', stroke=LINE, sw=1, rx=0))
            b.append(text(x + 36, y, v, 12.5, 600 if r == 0 else 400, INK, mono=(r > 0)))
    b.append(text(608, 236, '임계값 0.5: 확률 ≥ 0.5 → 1, < 0.5 → 0', 12, 500, RED))
    b.append(text(608, 256, '확률 0.5 ⇔ 로짓 0 ⇔ 분류 경계', 12, 500))
    svg('ch03_sigmoid', W, H, '시그모이드 함수',
        '로짓 −8부터 8까지의 시그모이드 곡선과 z=−5, 0, 5에서의 확률 0.0067, 0.5, 0.9933을 표로 보여 준다', ''.join(b))


# ── 그림 3-3. BCE 손실 곡선 ──────────────────────────────────────
def fig_bce():
    W, H = 760, 320
    b = [text(380, 26, '이진분류의 새 손실 — 정답 쪽 확률이 작을수록 손실이 가파르게 커진다', 15, 600)]
    for k, (y, c, lab, f) in enumerate([(1, RED, '정답 y = 1 일 때: −log p', lambda p: -math.log(p)),
                                         (0, '#2457A6', '정답 y = 0 일 때: −log(1 − p)', lambda p: -math.log(1 - p))]):
        ox = 60 + k * 360
        fr, X, Y = plot_frame(ox + 20, 62, 280, 180, (0, 1), (0, 5), [0, 0.2, 0.4, 0.6, 0.8, 1], [0, 1, 2, 3, 4, 5], '예측 확률 p = H(x)', '손실')
        b.append(fr)
        ps = [0.007 + i * (0.986 / 200) for i in range(201)]
        b.append(poly([(X(p), Y(min(f(p), 5))) for p in ps], c, 2.4))
        b.append(text(ox + 160, 52, lab, 13, 700, c))
        for p in (0.1, 0.5, 0.9):
            v = f(p)
            b.append(dot(X(p), Y(v), INK, 4))
            right = p < 0.8
            b.append(text(X(p) + (9 if right else -9), Y(v) - 12, f'{v:.3f}', 11.5, 700, c, 'start' if right else 'end', mono=True))
    b.append(text(380, 306, '두 식을 합치면 C = −y·log p − (1 − y)·log(1 − p). 정답이 1이면 앞 항만, 0이면 뒤 항만 남는다', 12.5, 500))
    svg('ch03_bce', W, H, 'BCE 손실 곡선',
        '정답이 1일 때 −log p, 정답이 0일 때 −log(1−p) 곡선. p=0.1, 0.5, 0.9에서 손실 2.303, 0.693, 0.105', ''.join(b))


# ── 그림 3-4. 로지스틱 분류의 학습과 예측 경로 ─────────────────────
def fig_logistic_flow():
    W, H = 760, 270
    b = [text(380, 26, '모델은 로짓까지만 — 학습과 예측에서 시그모이드를 쓰는 곳이 다르다', 15, 600)]
    b.append(box(20, 70, 110, 56, fill=GRAY)); b.append(text(75, 94, '입력 X', 14, 700)); b.append(text(75, 113, '[N, 특성 수]', 11.5, 400, '#4A4F5A', mono=True))
    b.append(arrow(132, 98, 168, 98))
    b.append(box(170, 70, 150, 56, fill=SOFT, stroke=PUR, sw=2)); b.append(text(245, 94, 'nn.Linear(n, 1)', 13, 700, PUR, mono=True)); b.append(text(245, 113, 'z = XWᵀ + b', 12, 400, INK, mono=True))
    b.append(f'<line x1="322" y1="98" x2="340" y2="98" stroke="{INK}" stroke-width="1.6"/>'); b.append(arrow(340, 98, 368, 72)); b.append(arrow(340, 98, 368, 132)); b.append(text(330, 90, 'z', 13, 700, PUR))
    # 학습 경로
    b.append(box(370, 46, 200, 48, fill='#FFF6D6', stroke=INK)); b.append(text(470, 67, '학습: BCEWithLogitsLoss(z, y)', 12.5, 700, INK, mono=False))
    b.append(text(470, 85, '시그모이드 + BCE를 한 번에', 11.5, 400, '#4A4F5A'))
    b.append(arrow(572, 70, 610, 70)); b.append(text(680, 74, 'backward → step', 12.5, 600, RED, mono=True))
    # 예측 경로
    b.append(box(370, 110, 200, 48, fill=TSOFT, stroke=TEAL, sw=2)); b.append(text(470, 131, '예측: torch.sigmoid(z)', 12.5, 700, TEAL, mono=True))
    b.append(text(470, 149, '확률 p (0 ~ 1)', 11.5, 400, '#4A4F5A'))
    b.append(arrow(572, 134, 610, 134)); b.append(box(612, 110, 130, 48, fill='#FFFFFF', stroke=TEAL)); b.append(text(677, 131, 'p ≥ 0.5 → 1', 12.5, 700, INK, mono=True)); b.append(text(677, 149, 'p < 0.5 → 0', 12.5, 700, INK, mono=True))
    b.append(text(380, 196, '로짓(logit): 확률로 바꾸기 전의 점수, 범위 −∞ ~ +∞. 로짓 0 ⇔ 확률 0.5', 12.5, 500))
    b.append(text(380, 220, '모델 안에 nn.Sigmoid()를 넣고 BCELoss를 써도 같은 계산이지만, 로짓이 극단이면 손실이 100에서 잘린다(추가 확인 03)', 12, 400, '#4A4F5A'))
    b.append(text(380, 244, '다중분류도 같다: 학습은 CrossEntropyLoss(Z, y), 확률은 torch.softmax(Z, dim=1), 답은 argmax', 12, 400, '#4A4F5A'))
    svg('ch03_logistic_flow', W, H, '로지스틱 분류의 학습과 예측 경로',
        '선형 계층이 로짓 z를 내고, 학습에서는 z를 BCEWithLogitsLoss에 바로 넣으며, 예측에서는 sigmoid로 확률을 만든 뒤 0.5와 비교한다', ''.join(b))


# ── 그림 3-5. 혼동행렬과 지표 ────────────────────────────────────
def fig_confusion():
    W, H = 760, 300
    TN, FP, FN, TP = 21, 7, 3, 30
    b = [text(380, 26, '혼동행렬 — 실습 3-2 심장질환 테스트 데이터 61명 (임계값 0.5)', 15, 600)]
    x0, y0, c = 150, 70, 96
    b.append(text(x0 + c, y0 - 18, '예측', 13, 700))
    b.append(text(x0 + c / 2, y0 - 2, '0 (없음)', 12, 500)); b.append(text(x0 + 1.5 * c, y0 - 2, '1 (있음)', 12, 500))
    b.append(f'<text x="{x0 - 70}" y="{y0 + c}" font-size="13" font-weight="700" fill="{INK}" text-anchor="middle" transform="rotate(-90 {x0 - 70} {y0 + c})">실제</text>')
    b.append(text(x0 - 10, y0 + c / 2 + 4, '0 (없음)', 12, 500, INK, 'end')); b.append(text(x0 - 10, y0 + 1.5 * c + 4, '1 (있음)', 12, 500, INK, 'end'))
    cells = [((0, 0), 'TN', TN, TSOFT, TEAL), ((0, 1), 'FP', FP, '#FBE9E7', RED), ((1, 0), 'FN', FN, '#FBE9E7', RED), ((1, 1), 'TP', TP, TSOFT, TEAL)]
    for (r, cc), lab, v, f, s in cells:
        x, y = x0 + cc * c, y0 + r * c
        b.append(f'<rect x="{x}" y="{y + 8}" width="{c}" height="{c}" fill="{f}" stroke="{s}" stroke-width="1.6"/>')
        b.append(text(x + c / 2, y + 44, lab, 14, 700, s)); b.append(text(x + c / 2, y + 74, str(v), 22, 700, INK, mono=True))
    n = TN + FP + FN + TP
    acc, pre, rec = (TP + TN) / n, TP / (TP + FP), TP / (TP + FN)
    f1 = 2 * pre * rec / (pre + rec)
    lines = [('Accuracy', '(TP + TN) / 전체', f'{TP + TN}/{n} = {acc:.4f}', '전체 중 맞힌 비율'),
             ('Precision', 'TP / (TP + FP)', f'{TP}/{TP + FP} = {pre:.4f}', '"있음"이라 한 것 중 진짜'),
             ('Recall', 'TP / (TP + FN)', f'{TP}/{TP + FN} = {rec:.4f}', '진짜 환자 중 찾아낸 비율'),
             ('F1', '2PR / (P + R)', f'{f1:.4f}', 'Precision과 Recall의 조화평균')]
    for i, (a, f, v, m) in enumerate(lines):
        y = 92 + i * 46
        b.append(text(420, y, a, 13.5, 700, PUR, 'start')); b.append(text(520, y, f, 12.5, 500, INK, 'start', mono=True))
        b.append(text(420, y + 19, v, 12.5, 600, INK, 'start', mono=True)); b.append(text(560, y + 19, m, 12, 400, '#4A4F5A', 'start'))
    b.append(text(380, 286, '대각선(TN, TP)은 맞힌 수, 대각선 밖(FP, FN)은 틀린 수. 의료 선별에서는 환자를 놓치는 FN이 특히 비싸다', 12.5, 500, RED))
    svg('ch03_confusion', W, H, '혼동행렬과 평가지표',
        f'실제 0·1과 예측 0·1의 네 칸 TN {TN}, FP {FP}, FN {FN}, TP {TP}와 정확도 {acc:.4f}, 정밀도 {pre:.4f}, 재현율 {rec:.4f}, F1 {f1:.4f}', ''.join(b))


# ── 그림 3-6. 소프트맥스와 교차엔트로피 ───────────────────────────
def fig_softmax():
    W, H = 760, 300
    s = [2.0, 1.0, 0.1]
    e = [math.exp(v) for v in s]
    p = [v / sum(e) for v in e]
    b = [text(380, 26, '소프트맥스 → 원핫 정답과 비교 → 교차엔트로피 (강의 예 [2.0, 1.0, 0.1])', 15, 600)]
    heads = [('로짓 Z', s, '{:.1f}', GRAY), ('e^z', e, '{:.3f}', '#FFFFFF'), ('확률 S = e^z / Σ', p, '{:.4f}', TSOFT), ('정답 L (원핫)', [1, 0, 0], '{}', SOFT)]
    xs = [40, 190, 340, 560]
    for i, (h, vals, fmt, f) in enumerate(heads):
        b.append(text(xs[i] + 55, 62, h, 12.5, 700))
        for r, v in enumerate(vals):
            b.append(box(xs[i], 74 + r * 40, 110, 34, fill=f, stroke=LINE, sw=1, rx=4))
            b.append(text(xs[i] + 55, 96 + r * 40, ('ABC'[r] + '  ' if i == 0 else '') + fmt.format(v), 13.5, 600, INK, mono=True))
        if i < 2:
            b.append(arrow(xs[i] + 112, 131, xs[i + 1] - 2, 131))
    b.append(text(xs[1] + 55, 210, f'Σ = {sum(e):.3f}', 12.5, 600, INK, mono=True))
    b.append(text(xs[2] + 55, 210, f'합 = {sum(p):.4f}', 12.5, 600, TEAL, mono=True))
    b.append(f'<path d="M 452 91 C 500 70, 520 70, 558 91" fill="none" stroke="{RED}" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#ah)" marker-start="url(#ah)"/>')
    b.append(text(380, 244, f'D(S, L) = −Σ Lᵢ log Sᵢ = −(1·log {p[0]:.4f} + 0 + 0) = {-math.log(p[0]):.4f}', 14, 700, RED, mono=True))
    b.append(text(380, 270, '원핫 정답이 1인 칸 하나만 남는다 → 정답 클래스 확률의 −log', 12.5, 500))
    b.append(text(380, 290, '강의 슬라이드는 확률을 0.7 · 0.2 · 0.1로 반올림해 −log 0.7 ≈ 0.357로 계산한다', 12, 400, '#4A4F5A'))
    svg('ch03_softmax', W, H, '소프트맥스와 교차엔트로피',
        f'로짓 2.0, 1.0, 0.1을 지수화해 합으로 나누면 확률 {p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}. 정답이 A이면 교차엔트로피는 {-math.log(p[0]):.4f}', ''.join(b))


for f in (fig_overview, fig_sigmoid, fig_bce, fig_logistic_flow, fig_confusion, fig_softmax):
    f()
print('생성:', sorted(p.name for p in OUT.glob('ch03_[a-z]*.svg')))
