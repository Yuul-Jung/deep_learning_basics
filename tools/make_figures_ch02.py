"""2장 그림(SVG) 생성기 — figures/ch02_*.svg

곡선·경로의 좌표는 이 스크립트가 식으로 계산한다(손으로 찍은 점 없음).
"""
from svgkit import *  # noqa: F401,F403




# ── 그림 2-1. 2장 큰 그림 ───────────────────────────────────────
def fig_overview():
    W, H = 760, 360
    b = [text(380, 28, '2장의 흐름 — 직선 하나를 데이터에 맞추는 세 단계', 16, 600)]
    steps = [('가설', 'H(x) = Wx + b', '직선으로 예측', PUR), ('손실', 'cost(W, b) = MSE', '틀린 정도를 숫자로', TEAL),
             ('경사하강법', 'W := W − α·∂cost/∂W', '손실이 줄어드는 쪽으로', '#B3261E'), ('여러 입력', 'H(X) = XW + b', '행렬곱 한 번에', INK)]
    xs = [16, 202, 388, 574]
    for i, (t, f, s, c) in enumerate(steps):
        b.append(box(xs[i], 52, 170, 92, fill='#FFFFFF', stroke=c, sw=2))
        b.append(text(xs[i] + 85, 78, t, 15, 700, c))
        b.append(text(xs[i] + 85, 104, f, 12.5, 500, INK, mono=True))
        b.append(text(xs[i] + 85, 128, s, 12.5, 400, '#4A4F5A'))
        if i < 3:
            b.append(arrow(xs[i] + 172, 98, xs[i + 1] - 2, 98))
    # 반복 고리
    b.append(f'<path d="M 473 146 C 473 178, 287 178, 287 146" fill="none" stroke="#B3261E" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#ah)"/>')
    b.append(text(380, 190, '손실이 충분히 작아질 때까지 반복 (epoch)', 12.5, 500, '#B3261E'))
    # 강의·실습 대응
    b.append(text(380, 226, '강의와 실습의 대응', 13.5, 600))
    rows = [('강의 2-1 가설과 손실함수', '실습 2-1 순전파와 손실 · 2-2 nn.Linear'),
            ('강의 2-2 손실 최소화 알고리즘', '실습 2-3 비용 최소화 · 2-4 실제 데이터(물고기)'),
            ('강의 2-3 여러 입력의 선형회귀', '실습 2-5 다중 선형회귀 · 2-6 주택가격')]
    for i, (l, r) in enumerate(rows):
        y = 246 + i * 36
        b.append(box(40, y, 300, 28, fill=SOFT, stroke=PUR, sw=1, rx=5)); b.append(text(190, y + 19, l, 12.5, 600, PUR))
        b.append(arrow(342, y + 14, 386, y + 14))
        b.append(box(390, y, 330, 28, fill=TSOFT, stroke=TEAL, sw=1, rx=5)); b.append(text(555, y + 19, r, 12.5, 500, INK))
    svg('ch02_overview', W, H, '2장 큰 그림',
        '가설, 손실, 경사하강법, 여러 입력의 네 단계와 강의 세 부분이 실습 2-1부터 2-6까지 어떻게 대응하는지 보여 준다', ''.join(b))


# ── 그림 2-2. 세 가설과 오차 ────────────────────────────────────
def fig_hypotheses():
    W, H = 760, 310
    b = [text(380, 28, '같은 세 점, 다른 세 직선 — 오차(세로선)가 작은 직선이 좋은 가설', 15, 600)]
    pts = [(1, 1), (2, 2), (3, 3)]
    lines = [(1.0, 0.0, PUR, 'H(x) = 1x + 0'), (0.5, 2.0, '#C77700', 'H(x) = 0.5x + 2'), (0.5, -0.5, '#B3261E', 'H(x) = 0.5x − 0.5')]
    for k, (w_, b_, c, lab) in enumerate(lines):
        ox = 40 + k * 240
        fr, X, Y = plot_frame(ox + 30, 60, 170, 160, (0, 3.5), (-1, 4), [0, 1, 2, 3], [0, 1, 2, 3], '', '')
        b.append(fr)
        b.append(poly([(X(0), Y(b_)), (X(3.5), Y(w_ * 3.5 + b_))], c, 2.2))
        for x, y in pts:
            yh = w_ * x + b_
            if abs(yh - y) > 1e-9:
                b.append(f'<line x1="{X(x):.1f}" y1="{Y(y):.1f}" x2="{X(x):.1f}" y2="{Y(yh):.1f}" stroke="{c}" stroke-width="1.6" stroke-dasharray="3 3"/>')
            b.append(dot(X(x), Y(y), INK, 4.5))
        mse = sum((w_ * x + b_ - y) ** 2 for x, y in pts) / 3
        b.append(text(ox + 115, 266, lab, 12.5, 700, c, mono=True))
        b.append(text(ox + 115, 290, f'MSE = {mse:.4g}', 12.5, 500, INK, mono=True))
    svg('ch02_hypotheses', W, H, '세 가설과 오차',
        '점 (1,1),(2,2),(3,3)에 대해 1x+0은 MSE 0, 0.5x+2는 1.1667, 0.5x−0.5는 2.4167', ''.join(b))


# ── 그림 2-3. 비용 곡선과 경사하강 경로 ──────────────────────────
def fig_cost_path():
    W, H = 760, 330
    b = [text(380, 28, 'cost(W) = 14/3 · (W − 1)²  위에서 W = 5 → 1 로 내려가는 경사하강법 (α = 0.1)', 14.5, 600)]
    fr, X, Y = plot_frame(90, 56, 420, 210, (-3, 5.5), (0, 80), [-3, -2, -1, 0, 1, 2, 3, 4, 5], [0, 20, 40, 60, 80], '가중치 W', 'cost')
    b.append(fr)
    cost = lambda w: 14 / 3 * (w - 1) ** 2
    curve = [(X(-3 + i * 0.05), Y(cost(-3 + i * 0.05))) for i in range(int(8.5 / 0.05) + 1) if cost(-3 + i * 0.05) <= 80]
    b.append(poly(curve, TEAL, 2.2))
    # 경로: dcost/dW = 28/3 (W-1)  (실습 2-3 의 2/m Σ 식과 같은 값)
    w, path = 5.0, []
    for _ in range(6):
        path.append(w); w = w - 0.1 * 28 / 3 * (w - 1)
    for i, wv in enumerate(path):
        b.append(dot(X(wv), Y(cost(wv)), '#B3261E', 5, '#FFFFFF' if i else '#B3261E'))
        if i:
            pw = path[i - 1]
            b.append(f'<line x1="{X(pw):.1f}" y1="{Y(cost(pw)):.1f}" x2="{X(wv):.1f}" y2="{Y(cost(wv)):.1f}" stroke="#B3261E" stroke-width="1.4" marker-end="url(#ah)"/>')
    b.append(dot(X(1), Y(0), PUR, 6)); b.append(text(X(-0.9), Y(14), '최저점 W = 1', 12, 700, PUR))
    b.append(f'<line x1="{X(-0.4):.1f}" y1="{Y(11):.1f}" x2="{X(0.85):.1f}" y2="{Y(1.5):.1f}" stroke="{PUR}" stroke-width="1"/>')
    # 오른쪽 표
    b.append(text(540, 70, '반복', 12.5, 700, INK, 'start')); b.append(text(600, 70, 'W', 12.5, 700, INK, 'start')); b.append(text(680, 70, 'cost', 12.5, 700, INK, 'start'))
    for i, wv in enumerate(path):
        y = 94 + i * 24
        b.append(text(548, y, str(i), 12.5, 400, INK, 'start', mono=True))
        b.append(text(600, y, f'{wv:.4f}', 12.5, 400, INK, 'start', mono=True))
        b.append(text(680, y, f'{cost(wv):.3f}', 12.5, 400, INK, 'start', mono=True))
    b.append(text(540, 254, '한 번 갈 때마다 W − 1 이', 12, 400, '#4A4F5A', 'start'))
    b.append(text(540, 272, '(1 − 0.1 × 28/3) ≈ 0.067 배로 줄어든다', 12, 400, '#4A4F5A', 'start'))
    b.append(text(380, 318, '기울기 = 28/3 · (W − 1): W > 1 이면 양수 → W 를 줄이고, W < 1 이면 음수 → W 를 늘린다', 12.5, 500))
    svg('ch02_cost_path', W, H, '비용 곡선과 경사하강 경로',
        'W=5에서 시작해 학습률 0.1로 한 번에 1.2667, 1.0178 순으로 최저점 W=1에 다가가는 경로', ''.join(b))


# ── 그림 2-4. 학습률 비교 ───────────────────────────────────────
def fig_lr():
    W, H = 760, 296
    b = [text(380, 28, '학습률 α 만 바꿨을 때 W 의 움직임 (W = 5 에서 시작, 처음 10번)', 15, 600)]
    cases = [(0.01, '너무 작음 · 느림', TEAL), (0.10, '적절함 · 빠른 수렴', PUR), (0.30, '너무 큼 · 발산', '#B3261E')]
    for k, (a, lab, c) in enumerate(cases):
        ox = 40 + k * 240
        ys = [5.0]
        for _ in range(10):
            ys.append(ys[-1] - a * 28 / 3 * (ys[-1] - 1))
        lo, hi = (-12, 14) if a == 0.30 else (0, 6)
        ticks = [-10, 0, 10] if a == 0.30 else [0, 1, 3, 5]
        fr, X, Y = plot_frame(ox + 30, 52, 180, 150, (0, 10), (lo, hi), [0, 5, 10], ticks, '반복 횟수', '')
        b.append(fr)
        b.append(f'<line x1="{X(0)}" y1="{Y(1):.1f}" x2="{X(10)}" y2="{Y(1):.1f}" stroke="{PUR}" stroke-dasharray="4 3" stroke-width="1"/>')
        clipped = [(X(i), Y(max(lo, min(hi, v)))) for i, v in enumerate(ys)]
        b.append(poly(clipped, c, 2))
        for x, y in clipped: b.append(dot(x, y, c, 3))
        b.append(text(ox + 120, 258, f'α = {a:.2f}  ·  {lab}', 12.5, 700, c))
    b.append(text(380, 282, '점선은 정답 W = 1. α > 0.214 이면 W 가 1 을 지나칠 때마다 더 멀리 튕겨 나간다 (그림 밖 값은 테두리에 표시)', 12, 400, '#4A4F5A'))
    svg('ch02_lr', W, H, '학습률 비교',
        '학습률 0.01은 천천히, 0.1은 빠르게 수렴하고, 0.3은 1을 넘나들며 점점 멀어져 발산한다', ''.join(b))


# ── 그림 2-5. 학습 반복 한 바퀴 ─────────────────────────────────
def fig_loop():
    W, H = 760, 220
    b = [text(380, 28, '학습 반복 한 바퀴 — 2장부터 5장까지 같은 다섯 줄', 16, 600)]
    st = [('순전파', 'prediction = model(x)', PUR), ('손실', 'loss = loss_function(prediction, y)', TEAL),
          ('기울기 비우기', 'optimizer.zero_grad()', INK), ('역전파', 'loss.backward()', '#B3261E'), ('갱신', 'optimizer.step()', '#1E6B34')]
    xs = [12, 162, 312, 462, 612]
    for i, (t, c_, col) in enumerate(st):
        b.append(box(xs[i], 56, 138, 84, fill='#FFFFFF', stroke=col, sw=2))
        b.append(text(xs[i] + 69, 84, f'{i + 1}. {t}', 13.5, 700, col))
        lines = {'prediction = model(x)': ['prediction =', 'model(x)'], 'loss = loss_function(prediction, y)': ['loss = loss_function(', 'prediction, y)']}.get(c_, [c_])
        for j, ln in enumerate(lines):
            b.append(text(xs[i] + 69, 112 + j * 16, ln, 10.5, 500, INK, mono=True))
        if i < 4: b.append(arrow(xs[i] + 140, 98, xs[i + 1] - 2, 98))
    b.append(f'<path d="M 681 142 C 681 196, 81 196, 81 142" fill="none" stroke="{INK}" stroke-width="1.4" stroke-dasharray="5 4" marker-end="url(#ah)"/>')
    b.append(text(380, 210, 'epochs 번 반복 · 실습 2-1은 zero_grad 를 맨 앞에, 2-2부터는 역전파 바로 앞에 둔다 (결과는 같음)', 12, 400, '#4A4F5A'))
    svg('ch02_loop', W, H, '학습 반복',
        '순전파, 손실, 기울기 비우기, 역전파, 갱신의 다섯 단계를 epoch 수만큼 반복한다', ''.join(b))


# ── 그림 2-6. 행렬로 쓴 가설 ────────────────────────────────────
def fig_matrix():
    W, H = 760, 300
    b = [text(380, 28, 'H(X) = XW — 학생 5명을 행렬곱 한 번으로', 16, 600)]
    data = [[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]]
    c = 40
    b.append(text(100, 58, 'X  [5, 3]', 13, 700, TEAL))
    b.append(grid(40, 68, data, cell=c, fill=TSOFT, stroke=TEAL, size=13, hl={(0, 0): YEL, (0, 1): YEL, (0, 2): YEL}))
    b.append(text(182, 174, '·', 26, 700))
    b.append(text(220, 58, 'W  [3, 1]', 13, 700, PUR))
    b.append(grid(200, 108, [['w1'], ['w2'], ['w3']], cell=c, fill=SOFT, stroke=PUR, size=13, hl={(0, 0): YEL, (1, 0): YEL, (2, 0): YEL}))
    b.append(text(264, 174, '=', 24, 700))
    b.append(text(372, 58, 'H(X)  [5, 1]', 13, 700, INK))
    rows = ['73w1 + 80w2 + 75w3', '93w1 + 88w2 + 93w3', '89w1 + 91w2 + 90w3', '96w1 + 98w2 + 100w3', '73w1 + 66w2 + 70w3']
    for r, v in enumerate(rows):
        f = YEL if r == 0 else '#FFFFFF'
        b.append(f'<rect x="286" y="{68 + r * c}" width="172" height="{c}" fill="{f}" stroke="{INK}" stroke-width="1.2"/>')
        b.append(text(372, 68 + r * c + 25, v, 12, 500, INK, mono=True))
    b.append(box(490, 68, 250, 200, fill='#FBFBF9', stroke=LINE, sw=1))
    b.append(text(615, 94, '모양 규칙', 13.5, 700))
    for i, (t, mono) in enumerate([('[n, 3] · [3, 1] → [n, 1]', True), ('[n, 3] · [3, 2] → [n, 2]', True), ('', False),
                                   ('앞 행렬의 열 수 = 뒤 행렬의 행 수', False), ('결과 = [앞의 행 수, 뒤의 열 수]', False)]):
        if t: b.append(text(615, 124 + i * 26, t, 12.5, 500 if mono else 400, INK, mono=mono))
    b.append(text(250, 292, '노란 칸: 첫 학생의 세 점수와 가중치 세 개를 곱해 더한 값이 첫 예측', 12, 400, '#4A4F5A'))
    svg('ch02_matrix', W, H, '행렬로 쓴 가설',
        '[5,3] 입력 행렬과 [3,1] 가중치의 행렬곱이 학생 다섯 명의 예측 [5,1]을 한 번에 계산한다. [n,3]에 [3,2]를 곱하면 [n,2]', ''.join(b))


# ── 그림 2-7. 데이터 분리와 표준화 ──────────────────────────────
def fig_split_scale():
    W, H = 760, 250
    b = [text(380, 28, '평균·표준편차는 훈련 데이터에서만 구하고, 테스트에는 그 값을 그대로 쓴다', 15, 600)]
    b.append(box(20, 60, 180, 120, fill=GRAY, stroke=INK)); b.append(text(110, 110, '전체 데이터', 14, 700)); b.append(text(110, 134, '섞기 randperm', 12, 400, INK, mono=True))
    b.append(arrow(202, 96, 258, 80)); b.append(arrow(202, 144, 258, 164))
    b.append(box(260, 52, 170, 60, fill=TSOFT, stroke=TEAL)); b.append(text(345, 78, '훈련 데이터', 13.5, 700, TEAL)); b.append(text(345, 98, '75~80 %', 12, 400))
    b.append(box(260, 136, 170, 60, fill=SOFT, stroke=PUR)); b.append(text(345, 162, '테스트 데이터', 13.5, 700, PUR)); b.append(text(345, 182, '20~25 %', 12, 400))
    b.append(arrow(432, 82, 488, 82)); b.append(box(490, 52, 250, 60, fill='#FFF6D6', stroke=INK))
    b.append(text(615, 78, 'μ, σ 계산 → 표준화 → 학습', 13, 700)); b.append(text(615, 98, '(x − μ) / σ', 12.5, 500, INK, mono=True))
    b.append(arrow(432, 166, 488, 166)); b.append(box(490, 136, 250, 60, fill='#FFFFFF', stroke=INK))
    b.append(text(615, 162, '같은 μ, σ 로 표준화 → 평가만', 13, 700)); b.append(text(615, 182, '학습·통계 계산에 쓰지 않음', 12, 400))
    b.append(f'<line x1="615" y1="114" x2="615" y2="134" stroke="{INK}" stroke-width="1.6" marker-end="url(#ah)"/>')
    b.append(text(380, 228, '테스트로 μ, σ 를 다시 구하면 평가 정보가 학습에 새어 들어간다 (데이터 누출, Data Leakage)', 12.5, 500, '#B3261E'))
    svg('ch02_split_scale', W, H, '데이터 분리와 표준화',
        '데이터를 섞어 훈련과 테스트로 나누고, 훈련 데이터의 평균과 표준편차로 두 쪽을 모두 표준화한다', ''.join(b))


for f in (fig_overview, fig_hypotheses, fig_cost_path, fig_lr, fig_loop, fig_matrix, fig_split_scale):
    f()
print('생성:', sorted(p.name for p in OUT.glob('ch02_[a-z]*.svg')))
