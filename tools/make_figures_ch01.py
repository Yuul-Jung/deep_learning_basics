"""1장 그림(SVG) 생성기 — figures/ch01_*.svg 를 만든다.

그림 원본은 이 스크립트다. 글자·좌표를 고친 뒤 다시 실행하면 SVG가 새로 만들어진다.
색: 글자 #15171C, 보라(PyTorch) #4635B1, 청록(NumPy) #1F7A8C, 노랑 강조 #FFD24A, 선 #9AA0A8
"""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parent.parent / 'figures'
OUT.mkdir(exist_ok=True)
INK, PUR, TEAL, YEL, LINE, SOFT, TSOFT, GRAY = '#15171C', '#4635B1', '#1F7A8C', '#FFD24A', '#9AA0A8', '#EDEAFA', '#E3F1F3', '#F5F5F2'
FONT = "IBM Plex Sans KR, Apple SD Gothic Neo, Malgun Gothic, sans-serif"
MONO = "IBM Plex Mono, Consolas, monospace"


def svg(name, w, h, title, desc, body):
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
         f'aria-labelledby="{name}-t {name}-d" font-family="{FONT}">'
         f'<title id="{name}-t">{escape(title)}</title><desc id="{name}-d">{escape(desc)}</desc>'
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         f'<path d="M0 0L10 5L0 10z" fill="{INK}"/></marker></defs>'
         f'<rect x="0" y="0" width="{w}" height="{h}" fill="#FFFFFF"/>' + body + '</svg>')
    s = s.replace('id="ah"', f'id="ah-{name}"').replace('url(#ah)', f'url(#ah-{name})')
    (OUT / f'{name}.svg').write_text(s, encoding='utf-8')


def box(x, y, w, h, fill='#FFFFFF', stroke=INK, sw=1.5, rx=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def text(x, y, s, size=14, weight=400, fill=INK, anchor='middle', mono=False):
    fam = f' font-family="{MONO}"' if mono else ''
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{fam}>{escape(s)}</text>'


def arrow(x1, y1, x2, y2, color=INK, dash=False, both=False):
    d = ' stroke-dasharray="5 4"' if dash else ''
    st = ' marker-start="url(#ah)"' if both else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6"{d} marker-end="url(#ah)"{st}/>'


def grid(x, y, vals, cell=34, fill='#FFFFFF', stroke=INK, size=14, hl=None):
    """2차원 값 표를 칸으로 그린다. hl: {(r,c): 색}"""
    out = []
    for r, row in enumerate(vals):
        for c, v in enumerate(row):
            f = (hl or {}).get((r, c), fill)
            out.append(f'<rect x="{x + c * cell}" y="{y + r * cell}" width="{cell}" height="{cell}" fill="{f}" stroke="{stroke}" stroke-width="1.2"/>')
            out.append(text(x + c * cell + cell / 2, y + r * cell + cell / 2 + 5, str(v), size, mono=True))
    return ''.join(out)


# ── 그림 1. 1장 큰 그림 ─────────────────────────────────────────────
def fig_overview():
    W, H = 760, 330
    b = []
    b.append(text(380, 30, '1장의 흐름 — 숫자 목록이 학습할 수 있는 텐서가 되기까지', 16, 600))
    xs = [16, 164, 312, 460, 608]
    heads = [('Python 리스트', GRAY, INK), ('NumPy 배열', TSOFT, TEAL), ('PyTorch 텐서', SOFT, PUR),
             ('자동 미분', SOFT, PUR), ('2장 · 선형회귀', '#FFF6D6', INK)]
    subs = [['[1, 2, 3]', '+ 는 이어 붙이기'], ['ndarray', '원소별 수치 계산'], ['Tensor', 'dtype · device'],
            ['requires_grad', 'backward() → grad'], ['학습 반복', '예측 → 손실 → 갱신']]
    labs = ['1-2', '1-1 · 1-2', '1-1 · 1-3', '1-1', '다음 장']
    for i, x in enumerate(xs):
        t, f, s = heads[i]
        b.append(box(x, 62, 136, 112, fill=f, stroke=s, sw=2))
        b.append(text(x + 68, 92, t, 15, 700, s))
        b.append(text(x + 68, 124, subs[i][0], 13, 500, INK, mono=True))
        b.append(text(x + 68, 150, subs[i][1], 12.5, 400, INK))
        b.append(text(x + 68, 196, '실습 ' + labs[i] if i < 4 else labs[i], 12, 500, '#676D78'))
        if i < 4:
            b.append(arrow(x + 138, 118, xs[i + 1] - 2, 118, both=(i == 1)))
    # 아래 띠: 텐서를 다루는 도구들
    b.append(box(140, 222, 480, 88, fill='#FBFBF9', stroke=LINE, sw=1.2))
    b.append(text(380, 246, '배열·텐서를 다루는 공통 도구', 13.5, 600))
    tools = [('모양', 'shape · reshape · flatten'), ('고르기', '인덱싱 · 불리언 마스크'),
             ('계산', '* · @ · 브로드캐스팅'), ('묶기', 'cat · stack')]
    for i, (k, v) in enumerate(tools):
        x = 160 + (i % 2) * 272
        y = 272 + (i // 2) * 24
        b.append(text(x, y, k, 12.5, 700, PUR, 'start'))
        b.append(text(x + 46, y, v, 12.5, 400, INK, 'start'))
    svg('ch01_overview', W, H, '1장 큰 그림',
        'Python 리스트에서 NumPy 배열, PyTorch 텐서, 자동 미분을 거쳐 2장 선형회귀 학습으로 이어지는 흐름과 공통 도구', ''.join(b))


# ── 그림 2. 스칼라·벡터·행렬·텐서 ────────────────────────────────
def fig_dims():
    W, H = 760, 250
    b = [text(380, 28, '차원 수(ndim)와 모양(shape)', 16, 600)]
    # scalar
    b.append(grid(60, 92, [[7]]))
    b.append(text(77, 168, 'Scalar', 14, 700, PUR)); b.append(text(77, 190, 'ndim 0', 12.5, 400, INK, mono=True))
    b.append(text(77, 210, 'shape ()', 12.5, 400, INK, mono=True))
    # vector
    b.append(grid(170, 92, [[1, 2, 3]]))
    b.append(text(221, 168, 'Vector', 14, 700, PUR)); b.append(text(221, 190, 'ndim 1', 12.5, 400, INK, mono=True))
    b.append(text(221, 210, 'shape (3,)', 12.5, 400, INK, mono=True))
    # matrix
    b.append(grid(330, 75, [[1, 2], [3, 4]]))
    b.append(text(364, 168, 'Matrix', 14, 700, PUR)); b.append(text(364, 190, 'ndim 2', 12.5, 400, INK, mono=True))
    b.append(text(364, 210, 'shape (2, 2)', 12.5, 400, INK, mono=True))
    # 3D tensor: 두 장을 겹쳐 그림
    b.append(grid(512, 58, [[' '] * 4] * 3, cell=24, fill=SOFT, stroke=PUR, size=11))
    b.append(grid(530, 76, [['', '', '', '']] * 3, cell=24, fill='#FFFFFF', stroke=PUR, size=11))
    b.append(text(578, 168, '3차원 텐서', 14, 700, PUR)); b.append(text(578, 190, 'ndim 3', 12.5, 400, INK, mono=True))
    b.append(text(578, 210, 'shape (2, 3, 4)', 12.5, 400, INK, mono=True))
    b.append(text(380, 240, 'shape 의 숫자 개수 = ndim · 숫자를 모두 곱하면 원소 개수', 12.5, 400, '#4A4F5A'))
    svg('ch01_dims', W, H, '스칼라·벡터·행렬·3차원 텐서',
        '값 1개는 ndim 0, 한 줄은 ndim 1, 표는 ndim 2, 표를 여러 장 쌓으면 ndim 3이며 shape는 각 차원의 길이를 적은 것', ''.join(b))


# ── 그림 3. 원소별 곱과 행렬곱 ───────────────────────────────────
def fig_mul():
    W, H = 760, 290
    a, bb = [[1, 2], [3, 4]], [[5, 6], [7, 8]]
    b = [text(380, 28, '* 는 같은 자리끼리, @ 는 행 × 열', 16, 600)]
    # 왼쪽: 원소별
    b.append(text(190, 62, '원소별 곱셈  a * b', 14, 700, TEAL))
    b.append(grid(40, 84, a, hl={(0, 1): YEL})); b.append(text(126, 124, '*', 20, 700))
    b.append(grid(146, 84, bb, hl={(0, 1): YEL})); b.append(text(232, 124, '=', 20, 700))
    b.append(grid(252, 84, [[5, 12], [21, 32]], hl={(0, 1): YEL}))
    b.append(text(190, 186, '노란 칸: 2 × 6 = 12', 13, 400))
    b.append(text(190, 208, '모양이 같은 두 배열의 같은 위치끼리 곱함', 12.5, 400, '#4A4F5A'))
    # 오른쪽: 행렬곱
    b.append(text(570, 62, '행렬곱  a @ b', 14, 700, PUR))
    b.append(grid(420, 84, a, hl={(0, 0): YEL, (0, 1): YEL})); b.append(text(506, 124, '@', 18, 700))
    b.append(grid(526, 84, bb, hl={(0, 0): YEL, (1, 0): YEL})); b.append(text(612, 124, '=', 20, 700))
    b.append(grid(632, 84, [[19, 22], [43, 50]], hl={(0, 0): YEL}))
    b.append(text(570, 186, '노란 칸: 1×5 + 2×7 = 19', 13, 400))
    b.append(text(570, 208, '앞 행렬의 행과 뒤 행렬의 열을 곱해 더함', 12.5, 400, '#4A4F5A'))
    b.append(f'<line x1="380" y1="52" x2="380" y2="220" stroke="{LINE}" stroke-width="1"/>')
    b.append(box(120, 236, 520, 40, fill='#FBFBF9', stroke=LINE, sw=1))
    b.append(text(380, 261, '신경망의 한 층 = 입력 X @ 가중치 W + 편향 b  → 2장부터 계속 쓰는 계산', 13, 500))
    svg('ch01_mul', W, H, '원소별 곱셈과 행렬곱',
        '[[1,2],[3,4]] 과 [[5,6],[7,8]] 의 원소별 곱은 [[5,12],[21,32]], 행렬곱은 [[19,22],[43,50]]', ''.join(b))


# ── 그림 4. 브로드캐스팅 ─────────────────────────────────────────
def fig_broadcast():
    W, H = 760, 190
    b = [text(380, 28, '브로드캐스팅 — 작은 값을 큰 배열 모양에 맞춰 늘린다', 16, 600)]
    cell = 58
    vals = [[22.5, 31.0, 36.0]]
    b.append(grid(40, 66, vals, cell=cell)); b.append(text(234, 100, '+', 22, 700))
    b.append(grid(256, 66, [[1.5]], cell=cell, fill=YEL))
    b.append(grid(314, 66, [[1.5, 1.5]], cell=cell, fill='#FFF6D6', stroke=LINE))
    b.append(text(372, 146, '1.5 를 3칸으로 늘린 것처럼 계산', 12.5, 400, '#4A4F5A'))
    b.append(text(450, 100, '=', 22, 700))
    b.append(grid(474, 66, [[24.0, 32.5, 37.5]], cell=cell, fill=TSOFT))
    b.append(text(561, 146, '보정 후 온도', 12.5, 400, TEAL))
    b.append(text(380, 178, '실제로 복사본을 만들지는 않는다. 계산할 때만 모양을 맞춘다.', 12.5, 400, '#4A4F5A'))
    svg('ch01_broadcast', W, H, '브로드캐스팅',
        '온도 [22.5, 31, 36] 에 1.5 를 더하면 1.5 가 세 칸으로 늘어난 것처럼 계산되어 [24, 32.5, 37.5] 가 된다', ''.join(b))


# ── 그림 5. cat 과 stack ─────────────────────────────────────────
def fig_cat_stack():
    W, H = 760, 332
    A, B = [[1, 2, 3], [4, 5, 6]], [[10, 20, 30], [40, 50, 60]]
    c = 30
    b = [text(380, 28, 'cat 은 기존 차원으로 잇고, stack 은 새 차원을 만들어 쌓는다', 16, 600)]
    b.append(text(60, 64, 'a  (2, 3)', 13, 700, TEAL, 'start')); b.append(grid(60, 74, A, cell=c, size=12))
    b.append(text(60, 160, 'b  (2, 3)', 13, 700, PUR, 'start')); b.append(grid(60, 170, B, cell=c, fill=SOFT, size=12))
    # cat dim=0
    b.append(text(290, 64, 'cat(dim=0) → (4, 3)', 13, 700, INK, 'start'))
    b.append(grid(290, 74, A, cell=c, size=12)); b.append(grid(290, 134, B, cell=c, fill=SOFT, size=12))
    # cat dim=1
    b.append(text(450, 64, 'cat(dim=1) → (2, 6)', 13, 700, INK, 'start'))
    b.append(grid(450, 74, A, cell=c, size=12)); b.append(grid(540, 74, B, cell=c, fill=SOFT, size=12))
    # stack dim=0
    b.append(text(450, 166, 'stack(dim=0) → (2, 2, 3)', 13, 700, INK, 'start'))
    b.append(f'<rect x="440" y="176" width="220" height="92" rx="6" fill="none" stroke="{PUR}" stroke-dasharray="5 4" stroke-width="1.4"/>')
    b.append(text(495, 194, '[0]', 12, 600, TEAL, mono=True)); b.append(text(605, 194, '[1]', 12, 600, PUR, mono=True))
    b.append(grid(450, 200, A, cell=c, size=12)); b.append(grid(560, 200, B, cell=c, fill=SOFT, size=12))
    b.append(text(672, 214, '새로 생긴', 12.5, 400, '#4A4F5A', 'start'))
    b.append(text(672, 232, '첫 차원', 12.5, 400, '#4A4F5A', 'start'))
    b.append(text(672, 250, '크기 2', 12.5, 400, '#4A4F5A', 'start'))
    b.append(box(40, 280, 680, 40, fill='#FBFBF9', stroke=LINE, sw=1))
    b.append(text(380, 305, 'cat: 측정값을 한 표로 이어 붙일 때 · stack: 묶음을 구분한 채 겹쳐 둘 때 (배치 만들기)', 13, 500))
    svg('ch01_cat_stack', W, H, 'cat 과 stack 비교',
        '(2,3) 텐서 두 개를 cat dim=0 하면 (4,3), dim=1 하면 (2,6), stack dim=0 하면 (2,2,3) 이 된다', ''.join(b))


# ── 그림 6. 메모리 공유와 복사 ───────────────────────────────────
def fig_memory():
    W, H = 760, 280
    b = [text(380, 28, '변환할 때 메모리를 함께 쓰는가, 복사하는가', 16, 600)]
    # 왼쪽: 공유
    b.append(box(24, 50, 348, 210, fill='#FBFBF9', stroke=LINE, sw=1))
    b.append(text(198, 76, '메모리를 함께 씀 (한쪽을 바꾸면 둘 다 바뀜)', 13.5, 700, '#B3261E'))
    b.append(box(48, 96, 120, 48, fill=TSOFT, stroke=TEAL)); b.append(text(108, 125, 'numpy_data', 13, 600, TEAL, mono=True))
    b.append(box(228, 96, 120, 48, fill=SOFT, stroke=PUR)); b.append(text(288, 125, 'shared_tensor', 13, 600, PUR, mono=True))
    b.append(box(118, 182, 160, 44, fill='#FFFFFF', stroke=INK)); b.append(text(198, 209, '[[10, 20], [30, 40]]', 12.5, 500, INK, mono=True))
    b.append(arrow(108, 146, 168, 180)); b.append(arrow(288, 146, 228, 180))
    b.append(text(198, 246, 'torch.from_numpy(arr) · tensor.numpy()', 12.5, 500, INK, mono=True))
    # 오른쪽: 복사
    b.append(box(388, 50, 348, 210, fill='#FBFBF9', stroke=LINE, sw=1))
    b.append(text(562, 76, '새로 복사함 (서로 독립)', 13.5, 700, '#1E6B34'))
    b.append(box(412, 96, 120, 48, fill=TSOFT, stroke=TEAL)); b.append(text(472, 125, 'np_data', 13, 600, TEAL, mono=True))
    b.append(box(592, 96, 120, 48, fill=SOFT, stroke=PUR)); b.append(text(652, 125, 'tensor_data', 13, 600, PUR, mono=True))
    b.append(box(412, 182, 120, 44, fill='#FFFFFF', stroke=INK)); b.append(text(472, 209, '값 묶음 ①', 12.5, 500))
    b.append(box(592, 182, 120, 44, fill='#FFFFFF', stroke=INK)); b.append(text(652, 209, '값 묶음 ②', 12.5, 500))
    b.append(arrow(472, 146, 472, 180)); b.append(arrow(652, 146, 652, 180))
    b.append(text(562, 246, 'torch.tensor(arr) · .clone() · .copy()', 12.5, 500, INK, mono=True))
    svg('ch01_memory', W, H, '메모리 공유와 복사',
        'from_numpy 와 numpy() 는 같은 메모리를 가리켜 한쪽 변경이 다른 쪽에 보이고, torch.tensor·clone·copy 는 새 메모리를 만든다', ''.join(b))


# ── 그림 7. 자동 미분 ────────────────────────────────────────────
def fig_autograd():
    W, H = 760, 250
    b = [text(380, 28, '자동 미분 — 계산 과정을 기록했다가 거꾸로 따라가며 기울기를 구한다', 16, 600)]
    b.append(box(40, 92, 120, 56, fill=SOFT, stroke=PUR, sw=2))
    b.append(text(100, 116, 'x = 3.0', 14, 700, PUR, mono=True)); b.append(text(100, 136, 'requires_grad', 11.5, 400, INK, mono=True))
    b.append(box(250, 60, 120, 44, fill='#FFFFFF')); b.append(text(310, 87, 'x²  = 9', 13.5, 500, mono=True))
    b.append(box(250, 136, 120, 44, fill='#FFFFFF')); b.append(text(310, 163, '2x  = 6', 13.5, 500, mono=True))
    b.append(box(460, 92, 120, 56, fill='#FFF6D6', stroke=INK, sw=2)); b.append(text(520, 125, 'y = 15', 14, 700, mono=True))
    b.append(arrow(162, 112, 248, 84)); b.append(arrow(162, 128, 248, 156))
    b.append(arrow(372, 82, 458, 110)); b.append(arrow(372, 158, 458, 132))
    b.append(text(310, 210, '앞으로 계산 (순전파): 값을 구하며 과정을 기록', 12.5, 400, '#4A4F5A'))
    b.append(box(610, 92, 130, 56, fill=SOFT, stroke=PUR, sw=2))
    b.append(text(675, 116, 'x.grad = 8', 14, 700, PUR, mono=True)); b.append(text(675, 136, 'dy/dx = 2x + 2', 11.5, 400, INK, mono=True))
    b.append(arrow(582, 120, 608, 120, color=PUR))
    b.append(text(675, 172, 'y.backward():', 12.5, 600, PUR, mono=True))
    b.append(text(675, 192, '기록을 거꾸로 따라', 12.5, 400, PUR))
    b.append(text(675, 210, '기울기 계산', 12.5, 400, PUR))
    b.append(text(380, 238, '2장에서는 x 대신 가중치 W, y 대신 손실(cost)을 놓고 같은 일을 한다', 12.5, 500, INK))
    svg('ch01_autograd', W, H, '자동 미분',
        'x=3 에서 y=x²+2x=15 를 계산하고 backward 로 dy/dx=2x+2=8 을 얻는 과정', ''.join(b))


for f in (fig_overview, fig_dims, fig_mul, fig_broadcast, fig_cat_stack, fig_memory, fig_autograd):
    f()
print('생성:', sorted(p.name for p in OUT.glob('ch01_*.svg')))
