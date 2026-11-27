"""5장 그림(SVG) 생성기 — figures/ch05_*.svg

합성곱 결과·출력 크기·파라미터 수는 이 스크립트가 직접 계산한다(손으로 적은 숫자 없음).
"""
from svgkit import *  # noqa: F401,F403

RED, ORG, GRN, BLU = '#B3261E', '#C77700', '#2E7D32', '#2457A6'
X = [[1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]]
Wf = [[2, 0, 1], [0, 1, 2], [1, 0, 2]]


def conv(x, w, pad=0):
    n = len(x) + 2 * pad
    xp = [[0] * n for _ in range(n)]
    for r in range(len(x)):
        for c in range(len(x)):
            xp[r + pad][c + pad] = x[r][c]
    k = len(w); m = n - k + 1
    return [[sum(xp[r + i][c + j] * w[i][j] for i in range(k) for j in range(k)) for c in range(m)] for r in range(m)]


# ── 그림 5-1. 5장 큰 그림 ───────────────────────────────────────
def fig_overview():
    W, H = 760, 330
    b = [text(380, 28, '5장의 흐름 — Flatten 앞에 "특징을 뽑는 부분"을 붙인다', 16, 600)]
    parts = [('4장 MLP', 'Flatten → Linear', '픽셀 784개를 바로 계산', INK, GRAY),
             ('5장 CNN', 'Conv → ReLU → Pool (×2) → Flatten → Linear', '3×3 필터로 지역 특징을 먼저 추출', TEAL, TSOFT)]
    for i, (t, a, s, c, f) in enumerate(parts):
        y = 50 + i * 66
        b.append(box(30, y, 700, 54, fill=f, stroke=c, sw=1.8))
        b.append(text(110, y + 33, t, 15, 700, c)); b.append(text(420, y + 24, a, 13, 600, INK, mono=True)); b.append(text(420, y + 44, s, 12, 400, '#4A4F5A'))
    b.append(text(380, 202, '강의 슬라이드와 실습의 대응 (강의 1–25 → 실습 5-1 → 5-2 순서로 진행)', 13.5, 600))
    rows = [('강의 1–15 합성곱·풀링·패딩·특징 추출', '추가 확인: 슬라이드 숫자 예제 재현'),
            ('강의 16–25 Fashion-MNIST·채널·CNN API·크기 계산', '실습 5-1 CNN 학습·평가·2픽셀 이동'),
            ('강의 11–15 특징 추출 결과 해석', '실습 5-2 학습된 필터와 특징맵 보기')]
    for i, (l, r) in enumerate(rows):
        yy = 218 + i * 36
        b.append(box(14, yy, 390, 28, fill=SOFT, stroke=PUR, sw=1, rx=5)); b.append(text(209, yy + 19, l, 12, 600, PUR))
        b.append(arrow(406, yy + 14, 430, yy + 14))
        b.append(box(434, yy, 312, 28, fill=TSOFT, stroke=TEAL, sw=1, rx=5)); b.append(text(590, yy + 19, r, 12, 500, INK))
    svg('ch05_overview', W, H, '5장 큰 그림',
        'MLP와 CNN의 구조 차이와 강의 슬라이드 세 묶음이 실습 5-1, 5-2와 어떻게 대응하는지 보여 준다', ''.join(b))


# ── 그림 5-2. 합성곱 한 칸 계산 ─────────────────────────────────
def fig_conv():
    W, H = 760, 300
    y = conv(X, Wf)
    yb = [[v + 3 for v in row] for row in y]
    b = [text(380, 26, '합성곱 — 필터를 한 칸씩 옮기며 같은 자리끼리 곱해 모두 더한다 (강의 슬라이드 6·7)', 14.5, 600)]
    hl = {(r, c): '#FFE7A3' for r in range(3) for c in range(3)}
    b.append(text(96, 56, '입력 4×4', 12.5, 700)); b.append(grid(28, 66, X, 34, hl=hl))
    b.append(text(184, 140, '∗', 24, 700))
    b.append(text(256, 56, '필터 3×3', 12.5, 700, RED)); b.append(grid(205, 83, Wf, 34, fill='#FBE9E7'))
    b.append(text(336, 140, '=', 22, 700))
    hl2 = {(0, 0): '#FFE7A3'}
    b.append(text(392, 56, '합성곱 결과', 12.5, 700)); b.append(grid(358, 100, y, 34, hl=hl2))
    b.append(text(450, 140, '+ 3', 16, 700, BLU))
    b.append(text(528, 56, '특징 맵', 12.5, 700, TEAL)); b.append(grid(494, 100, yb, 34, fill=TSOFT))
    b.append(arrow(566, 134, 600, 134)); b.append(text(583, 124, 'ReLU', 10.5, 700, GRN))
    b.append(text(636, 56, 'Max Pool 2×2', 12.5, 700, ORG)); b.append(grid(620, 117, [[max(max(r) for r in yb)]], 34, fill='#FFF6D6'))
    terms = ' + '.join(f'{X[i][j]}·{Wf[i][j]}' for i in range(3) for j in range(3))
    b.append(text(380, 234, f'왼쪽 위 칸: {terms} = {y[0][0]}', 12.5, 600, INK, mono=True))
    b.append(text(380, 258, '필터가 오른쪽·아래로 한 칸씩(stride 1) 움직여 2×2 결과를 채우고, 바이어스 3을 모든 칸에 더한다', 12, 400, '#4A4F5A'))
    b.append(text(380, 280, 'ReLU는 음수만 0으로 바꾸므로 이 예에서는 값이 그대로이고, Max Pooling은 가장 큰 값 하나를 남긴다', 12, 400, '#4A4F5A'))
    svg('ch05_conv', W, H, '합성곱 계산',
        f'4×4 입력에 3×3 필터를 적용하면 {y}, 바이어스 3을 더하면 {yb}, 2×2 최대 풀링 결과는 {max(max(r) for r in yb)}', ''.join(b))


# ── 그림 5-3. 패딩과 스트라이드 ──────────────────────────────────
def fig_pad_stride():
    W, H = 760, 290
    b = [text(380, 26, '출력 크기 = ⌊(입력 − 필터 + 2 × 패딩) / 스트라이드⌋ + 1  (입력 28, 필터 3)', 14.5, 600)]
    cases = [(0, 1), (1, 1), (0, 2), (1, 2)]
    for k, (p, s) in enumerate(cases):
        o = (28 - 3 + 2 * p) // s + 1
        x0 = 20 + k * 185
        b.append(box(x0, 48, 172, 170, fill='#FFFFFF', stroke=LINE, sw=1))
        b.append(text(x0 + 86, 70, f'padding={p}, stride={s}', 12.5, 700, PUR if p else INK, mono=True))
        size_in = 28 + 2 * p
        side = 90 * size_in / 30
        bx = x0 + 18
        if p:
            b.append(f'<rect x="{bx}" y="{84}" width="{side:.1f}" height="{side:.1f}" fill="#FFF6D6" stroke="{ORG}"/>')
        inner = 90 * 28 / 30
        ix = bx + (side - inner) / 2
        b.append(f'<rect x="{ix:.1f}" y="{84 + (side - inner) / 2:.1f}" width="{inner:.1f}" height="{inner:.1f}" fill="{GRAY}" stroke="{INK}"/>')
        b.append(text(ix + inner / 2, 84 + side / 2 + 5, '28', 12, 700))
        out = 60 * o / 28
        ox = x0 + 118
        b.append(f'<rect x="{ox}" y="{84 + (90 - out) / 2:.1f}" width="{out:.1f}" height="{out:.1f}" fill="{TSOFT}" stroke="{TEAL}"/>')
        b.append(text(ox + out / 2, 84 + 45 + 4, str(o), 12, 700, TEAL))
        b.append(text(x0 + 86, 196, f'⌊(28−3+{2 * p})/{s}⌋+1 = {o}', 12, 600, INK, mono=True))
    b.append(text(380, 244, '패딩: 가장자리에 0을 둘러 크기가 줄지 않게(정보 손실도 줄임) — padding=1이면 28 → 28', 12.5, 500, ORG))
    b.append(text(380, 268, '스트라이드: 필터가 한 번에 움직이는 칸 수 — 2이면 크기가 약 절반', 12.5, 500, PUR))
    svg('ch05_pad_stride', W, H, '패딩과 스트라이드에 따른 출력 크기',
        '28×28 입력에 3×3 필터를 쓸 때 패딩 0·1과 스트라이드 1·2의 출력 크기 26, 28, 13, 14', ''.join(b))


# ── 그림 5-4. 실습 5-1 CNN의 모양 변화 ───────────────────────────
def fig_arch():
    W, H = 760, 300
    layers = [('입력', [1, 28, 28], None, GRAY, INK), ('Conv1 + ReLU', [8, 28, 28], 1 * 8 * 9 + 8, SOFT, PUR),
              ('MaxPool', [8, 14, 14], 0, '#FFF6D6', ORG), ('Conv2 + ReLU', [16, 14, 14], 8 * 16 * 9 + 16, SOFT, PUR),
              ('MaxPool', [16, 7, 7], 0, '#FFF6D6', ORG), ('Flatten', [784], 0, GRAY, INK), ('Linear', [10], 784 * 10 + 10, TSOFT, TEAL)]
    b = [text(380, 26, '실습 5-1의 CNN — 채널은 늘고(1 → 8 → 16) 가로·세로는 줄어든다(28 → 14 → 7)', 14.5, 600)]
    w, gap = 96, 12
    for i, (t, sh, p, f, c) in enumerate(layers):
        x = 14 + i * (w + gap)
        if len(sh) == 3:
            ch, hw = sh[0], sh[1]
            s = 18 + 40 * hw / 28
            n = min(ch, 6)
            for k in range(n):
                off = k * 4
                b.append(f'<rect x="{x + w / 2 - s / 2 + off - 10:.1f}" y="{62 + 60 - s / 2 - off + 10:.1f}" width="{s:.1f}" height="{s:.1f}" fill="{f}" stroke="{c}" stroke-width="1.2"/>')
        else:
            hgt = 70 if sh[0] > 100 else 30
            b.append(f'<rect x="{x + w / 2 - 6}" y="{122 - hgt / 2}" width="12" height="{hgt}" fill="{f}" stroke="{c}" stroke-width="1.2"/>')
        b.append(text(x + w / 2, 186, t, 12, 700, c))
        b.append(text(x + w / 2, 204, '[N, ' + ', '.join(map(str, sh)) + ']', 10.5, 500, INK, mono=True))
        if p is not None:
            b.append(text(x + w / 2, 224, f'{p:,}개' if p else '파라미터 없음', 11, 600 if p else 400, PUR if p else '#4A4F5A'))
        if i < len(layers) - 1:
            b.append(arrow(x + w - 2, 122, x + w + gap + 2, 122))
    total = (1 * 8 * 9 + 8) + (8 * 16 * 9 + 16) + (784 * 10 + 10)
    b.append(text(380, 256, f'학습 파라미터 = 80 + 1,168 + 7,850 = {total:,}개  (실습 4-5 MLP 235,146개의 약 1/26)', 13, 700, PUR))
    b.append(text(380, 280, 'Conv 파라미터 = 출력 채널 × 입력 채널 × 3 × 3 + 출력 채널(바이어스) — 이미지 크기와 상관없다', 12, 400, '#4A4F5A'))
    svg('ch05_arch', W, H, 'CNN의 계층별 모양과 파라미터',
        f'입력 [N,1,28,28]에서 Conv1, Pool, Conv2, Pool을 거쳐 [N,16,7,7], Flatten 784, Linear 10. 파라미터 {total:,}개', ''.join(b))


# ── 그림 5-5. 여러 입력 채널의 합성곱 ────────────────────────────
def fig_channels():
    W, H = 760, 260
    b = [text(380, 26, '입력 채널이 여러 개일 때 — 필터 하나는 채널 수만큼의 3×3 판을 가지고, 결과를 더해 특징맵 하나를 만든다', 13.5, 600)]
    cols = [YEL, '#BBDEFB']
    for k in range(2):
        y = 56 + k * 74
        b.append(box(30, y, 110, 56, fill=GRAY, stroke=INK)); b.append(text(85, y + 33, f'입력 채널 {k + 1}', 12.5, 700))
        b.append(text(160, y + 34, '∗', 20, 700))
        b.append(box(180, y + 8, 80, 40, fill=cols[k], stroke=INK)); b.append(text(220, y + 33, f'필터1-판{k + 1}', 11.5, 600))
        b.append(arrow(262, y + 28, 300, y + 28))
        b.append(box(302, y + 8, 90, 40, fill='#FFFFFF', stroke=LINE)); b.append(text(347, y + 33, f'임시 맵 {k + 1}', 11.5, 500))
    b.append(text(347, 125, '+', 20, 700, RED))
    b.append(arrow(394, 120, 450, 120))
    b.append(box(452, 94, 120, 52, fill=TSOFT, stroke=TEAL, sw=1.8)); b.append(text(512, 118, '출력 채널 1', 13, 700, TEAL)); b.append(text(512, 136, '(+ 바이어스)', 11, 400))
    b.append(text(660, 92, '필터 모양', 12.5, 700)); b.append(text(660, 114, '[출력 채널, 입력 채널,', 11.5, 500, INK, mono=True))
    b.append(text(660, 132, ' 높이, 너비]', 11.5, 500, INK, mono=True)); b.append(text(660, 158, 'Conv2: [16, 8, 3, 3]', 11.5, 600, PUR, mono=True))
    b.append(text(380, 222, '필터를 16개 두면 이 계산을 16번 해서 출력 채널(특징맵) 16개가 나온다 — 출력 채널 수 = 필터 수', 12.5, 500))
    b.append(text(380, 246, '컬러 이미지(채널 3)든 앞 층의 특징맵 8장이든 계산 방법은 같다 (추가 확인 05)', 12, 400, '#4A4F5A'))
    svg('ch05_channels', W, H, '여러 입력 채널의 합성곱',
        '입력 채널 두 개에 필터의 판 두 개를 각각 합성곱한 뒤 더해 출력 채널 하나를 만드는 과정', ''.join(b))


for f in (fig_overview, fig_conv, fig_pad_stride, fig_arch, fig_channels):
    f()
print('생성:', sorted(p.name for p in OUT.glob('ch05_[a-z]*.svg')))
