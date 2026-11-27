"""4장 그림(SVG) 생성기 — figures/ch04_*.svg

파라미터 수·배치 수는 이 스크립트가 계산한다(손으로 적은 숫자 없음).
"""
import math
from svgkit import *  # noqa: F401,F403

RED, ORG, GRN = '#B3261E', '#C77700', '#2E7D32'


# ── 그림 4-1. 4장 큰 그림 ───────────────────────────────────────
def fig_overview():
    W, H = 760, 380
    b = [text(380, 28, '4장의 흐름 — 은닉층을 쌓고, 큰 데이터를 배치로 나눠 학습·검증·평가한다', 15, 600)]
    steps = [('데이터 분리', 'train·val·test', GRAY, INK), ('DataLoader', '배치 64·shuffle', GRAY, INK),
             ('MLP', 'Linear + ReLU', SOFT, PUR), ('CrossEntropy', '+ Adam', SOFT, PUR),
             ('학습 · 검증', '매 Epoch 기록', TSOFT, TEAL), ('테스트', '마지막에 한 번', '#FFF6D6', INK), ('분석 · 예측', '혼동행렬 · 오답', '#FFFFFF', INK)]
    w, gap, y = 98, 10, 52
    for i, (t, s, f, c) in enumerate(steps):
        x = 12 + i * (w + gap)
        b.append(box(x, y, w, 62, fill=f, stroke=c, sw=1.6))
        b.append(text(x + w / 2, y + 26, t, 12.5, 700, c)); b.append(text(x + w / 2, y + 46, s, 10, 400, '#4A4F5A'))
        if i < len(steps) - 1:
            b.append(arrow(x + w + 1, y + 31, x + w + gap - 1, y + 31))
    b.append(text(380, 150, '3장과 같은 것: 로짓 → CrossEntropyLoss, 학습 반복 다섯 줄, argmax로 답 고르기', 12.5, 500, GRN))
    b.append(text(380, 172, '새로 생긴 것: 은닉층·ReLU, 배치·Epoch, 검증 데이터, Optimizer 비교, 과적합과 Dropout, 오답 분석', 12.5, 500, RED))
    b.append(text(380, 210, '강의 슬라이드와 실습의 대응 (강의 1–23 → 실습 4-1 … 4-6 순서로 진행)', 13.5, 600))
    rows = [('강의 1–6 딥러닝과 MLP 구조', '개념 (추가 확인: XOR, 층 합치기)'),
            ('강의 7–22 MNIST·데이터 분리·배치·학습·평가', '실습 4-1 MNIST MLP 전체 흐름'),
            ('강의 23 MLP의 한계 → CNN', '실습 4-5의 2픽셀 이동 실험 → 5장'),
            ('강의에 없는 심화', '실습 4-2 Optimizer · 4-3 Dropout · 4-4 오답 · 4-6 예측')]
    for i, (l, r) in enumerate(rows):
        yy = 226 + i * 38
        b.append(box(14, yy, 380, 30, fill=SOFT if i < 3 else '#FFFFFF', stroke=PUR, sw=1, rx=5)); b.append(text(204, yy + 20, l, 12, 600, PUR))
        b.append(arrow(396, yy + 15, 420, yy + 15))
        b.append(box(424, yy, 322, 30, fill=TSOFT, stroke=TEAL, sw=1, rx=5)); b.append(text(585, yy + 20, r, 12, 500, INK))
    svg('ch04_overview', W, H, '4장 큰 그림',
        '데이터 분리부터 분석까지 일곱 단계의 흐름과 강의 세 부분과 실습 여섯 개의 대응을 보여 준다', ''.join(b))


# ── 그림 4-2. MLP 구조와 파라미터 수 ─────────────────────────────
def fig_mlp():
    W, H = 760, 330
    layers = [('입력', '[N, 1, 28, 28]', None, GRAY, INK), ('Flatten', '[N, 784]', 0, GRAY, INK),
              ('Linear', '784 → 128', 784 * 128 + 128, SOFT, PUR), ('ReLU', '[N, 128]', 0, '#E8F5E9', GRN),
              ('Dropout(0.2)', '[N, 128]', 0, '#FFF6D6', ORG), ('Linear', '128 → 10', 128 * 10 + 10, SOFT, PUR), ('로짓', '[N, 10]', None, TSOFT, TEAL)]
    b = [text(380, 26, '실습 4-1의 MLP — 이미지를 784개 숫자로 펼친 뒤 두 번의 Linear로 로짓 10개를 만든다', 14.5, 600)]
    w, gap = 96, 12
    for i, (t, s, p, f, c) in enumerate(layers):
        x = 14 + i * (w + gap)
        b.append(box(x, 50, w, 70, fill=f, stroke=c, sw=1.8))
        b.append(text(x + w / 2, 78, t, 13, 700, c)); b.append(text(x + w / 2, 100, s, 11, 500, INK, mono=True))
        if p is not None:
            b.append(text(x + w / 2, 140, f'{p:,}개' if p else '파라미터 없음', 11.5, 600 if p else 400, PUR if p else '#4A4F5A'))
        if i < len(layers) - 1:
            b.append(arrow(x + w + 1, 85, x + w + gap - 1, 85))
    total = 784 * 128 + 128 + 128 * 10 + 10
    b.append(text(380, 170, f'학습 파라미터 = (784×128 + 128) + (128×10 + 10) = {784 * 128 + 128:,} + {128 * 10 + 10:,} = {total:,}개', 13, 700, PUR, mono=False))
    # ReLU 그래프
    fr, X, Y = plot_frame(70, 200, 200, 100, (-3, 3), (-1, 3), [-2, 0, 2], [0, 2], '', '')
    b.append(fr); b.append(poly([(X(-3), Y(0)), (X(0), Y(0)), (X(3), Y(3))], GRN, 2.4))
    b.append(text(170, 196, 'ReLU(x) = max(0, x)', 12.5, 700, GRN, mono=True))
    b.append(text(520, 220, 'ReLU가 없으면 Linear 두 개는 Linear 하나와 같다', 12.5, 700, RED))
    b.append(text(520, 242, 'W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)', 12.5, 500, INK, mono=True))
    b.append(text(520, 266, '음수를 0으로 자르는 꺾임이 있어야 층을 쌓는 의미가 생긴다', 12, 400, '#4A4F5A'))
    b.append(text(520, 288, '(추가 확인 01·02: XOR 문제와 두 층 합치기)', 11.5, 400, '#4A4F5A'))
    svg('ch04_mlp', W, H, 'MLP 구조와 파라미터 수',
        f'입력, Flatten, Linear 784→128, ReLU, Dropout, Linear 128→10, 로짓의 순서와 학습 파라미터 {total:,}개, ReLU 그래프', ''.join(b))


# ── 그림 4-3. Batch · Iteration · Epoch ─────────────────────────
def fig_batch():
    W, H = 760, 270
    n, bs, ep = 55_000, 64, 5
    it = math.ceil(n / bs); last = n - (it - 1) * bs
    b = [text(380, 26, f'학습 데이터 {n:,}개를 배치 {bs}개씩 — 한 Epoch에 Iteration {it}번, {ep} Epoch면 가중치 갱신 {it * ep:,}번', 14, 600)]
    x0, y0, cw = 30, 56, 700
    cells = 24
    for k in range(cells):
        x = x0 + k * cw / cells
        last_cell = k == cells - 1
        f = '#FFF6D6' if last_cell else (SOFT if k % 2 == 0 else '#FFFFFF')
        b.append(f'<rect x="{x:.1f}" y="{y0}" width="{cw / cells:.1f}" height="40" fill="{f}" stroke="{LINE}"/>')
    b.append(text(x0 + cw / cells / 2, y0 + 25, '#1', 10, 600, PUR)); b.append(text(x0 + 1.5 * cw / cells, y0 + 25, '#2', 10, 600, PUR))
    b.append(text(x0 + cw / 2, y0 + 25, '· · ·', 14, 700, '#4A4F5A'))
    b.append(text(x0 + cw - cw / cells / 2, y0 + 25, f'#{it}', 10, 600, ORG))
    b.append(text(x0 + cw / 2, y0 + 62, f'1 Epoch = 배치 {it}개 (꽉 찬 배치 {it - 1}개 × {bs}장 + 마지막 배치 {last}장)', 12.5, 600))
    rows = [('Batch', f'한 번에 모델에 넣는 묶음 ({bs}장)', '[64, 1, 28, 28] 이미지 + [64] 정답'),
            ('Iteration', '배치 하나로 순전파 → 손실 → 역전파 → 갱신 한 번', f'Epoch당 {it}번'),
            ('Epoch', '학습 데이터 전체를 한 번 다 보는 것', f'{ep}번 반복 → 갱신 {it * ep:,}번')]
    for i, (a, m, e) in enumerate(rows):
        y = 150 + i * 36
        b.append(box(30, y, 110, 28, fill=SOFT, stroke=PUR, sw=1, rx=4)); b.append(text(85, y + 19, a, 13, 700, PUR))
        b.append(text(156, y + 19, m, 12.5, 500, INK, 'start')); b.append(text(730, y + 19, e, 12, 600, TEAL, 'end', mono=False))
    svg('ch04_batch', W, H, 'Batch, Iteration, Epoch',
        f'5만 5천 개를 64개씩 나누면 Epoch당 {it}번 갱신하고 마지막 배치는 {last}장이다. 5 Epoch면 {it * ep}번 갱신한다', ''.join(b))


# ── 그림 4-4. 데이터 세 갈래 ─────────────────────────────────────
def fig_split():
    W, H = 760, 260
    b = [text(380, 26, 'MNIST 70,000장의 세 갈래 — 각자 맡은 일이 다르다', 15, 600)]
    parts = [('train', 55_000, '#C8E6C9', GRN, '가중치·편향을 학습', '학습 중 매 배치'),
             ('validation', 5_000, '#E1D5F5', PUR, '매 Epoch 상태 확인 · 모델/설정 고르기', '학습 중 매 Epoch'),
             ('test', 10_000, '#FFE0B2', ORG, '최종 성능 한 번 측정', '모든 선택이 끝난 뒤')]
    x, total, wdt = 30, 70_000, 700
    for name, n, f, c, role, when in parts:
        w = wdt * n / total
        b.append(f'<rect x="{x:.1f}" y="50" width="{w:.1f}" height="44" fill="{f}" stroke="{c}" stroke-width="1.6"/>')
        b.append(text(x + w / 2, 70, 'val' if name == 'validation' else name, 13, 700, c)); b.append(text(x + w / 2, 87, f'{n:,}', 12, 600, INK, mono=True))
        x += w
    b.append(text(30 + wdt * 30_000 / total, 112, '원래 훈련 60,000 → 55,000 + 5,000 (random_split, seed 42)', 11.5, 500, '#4A4F5A'))
    b.append(text(30 + wdt * 65_000 / total, 112, '원래 테스트 10,000', 11.5, 500, '#4A4F5A'))
    for i, (name, n, f, c, role, when) in enumerate(parts):
        y = 132 + i * 34
        b.append(box(30, y, 120, 26, fill=f, stroke=c, sw=1, rx=4)); b.append(text(90, y + 18, name, 12.5, 700, c))
        b.append(text(166, y + 18, role, 12.5, 500, INK, 'start')); b.append(text(730, y + 18, when, 12, 600, c, 'end'))
    b.append(text(380, 246, '테스트 결과를 보고 학습률·구조를 다시 고르면 테스트가 또 하나의 검증 데이터가 되어 버린다', 12.5, 500, RED))
    svg('ch04_split', W, H, '훈련·검증·테스트 데이터',
        '55,000개 훈련, 5,000개 검증, 10,000개 테스트의 비율과 각 데이터의 역할, 쓰는 시점', ''.join(b))


# ── 그림 4-5. Dropout: train() 과 eval() ─────────────────────────
def fig_dropout():
    W, H = 760, 290
    b = [text(380, 26, 'Dropout(p) — 학습 때만 은닉 출력 일부를 무작위로 0으로, 평가 때는 모두 사용', 14.5, 600)]
    for k, (title, drop, note) in enumerate([('model.train() — p = 0.3', True, '매번 다른 노드가 꺼짐 → 같은 입력도 출력이 달라짐'),
                                              ('model.eval() — Dropout 꺼짐', False, '모든 노드 사용 → 같은 입력이면 같은 출력')]):
        ox = 30 + k * 370
        b.append(text(ox + 165, 56, title, 13, 700, ORG if drop else TEAL))
        cols = [(ox + 30, 4), (ox + 130, 6), (ox + 230, 6), (ox + 310, 3)]
        pos = [[(cx, 90 + (150 - (n - 1) * 26) / 2 + j * 26) for j in range(n)] for cx, n in cols]
        off = {(1, 1), (1, 4), (2, 0), (2, 3)} if drop else set()   # 은닉 노드 12개 중 4개(약 30%)
        for L in range(3):
            for i, p in enumerate(pos[L]):
                for j, q in enumerate(pos[L + 1]):
                    if (L, i) in off or (L + 1, j) in off:
                        continue
                    b.append(f'<line x1="{p[0]}" y1="{p[1]:.1f}" x2="{q[0]}" y2="{q[1]:.1f}" stroke="#C9CCD3" stroke-width="1"/>')
        for L in range(4):
            for j, (x, y) in enumerate(pos[L]):
                dead = (L, j) in off
                fill = '#FFFFFF' if dead else (GRAY if L == 0 else (TSOFT if L == 3 else SOFT))
                b.append(f'<circle cx="{x}" cy="{y:.1f}" r="9" fill="{fill}" stroke="{LINE if dead else INK}" stroke-width="1.4"{" stroke-dasharray=\"3 2\"" if dead else ""}/>')
                if dead:
                    b.append(text(x, y + 4, '0', 10, 700, RED))
        b.append(text(ox + 165, 262, note, 12, 500, INK))
    b.append(text(380, 284, 'PyTorch는 학습 때 남은 출력을 1/(1−p)배 키워, 평가 때 모두 켜도 출력의 크기가 맞도록 한다', 11.5, 400, '#4A4F5A'))
    svg('ch04_dropout', W, H, 'Dropout의 학습과 평가',
        '왼쪽은 train 모드에서 은닉 노드 일부가 0이 된 신경망, 오른쪽은 eval 모드에서 모든 노드가 연결된 신경망', ''.join(b))


for f in (fig_overview, fig_mlp, fig_batch, fig_split, fig_dropout):
    f()
print('생성:', sorted(p.name for p in OUT.glob('ch04_[a-z]*.svg')))
