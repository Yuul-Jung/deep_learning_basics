"""부록 A 그림(SVG) 생성기 — figures/appA_*.svg"""
from svgkit import *  # noqa: F401,F403

RED, ORG, GRN, BLU = '#B3261E', '#C77700', '#2E7D32', '#2457A6'


# ── 그림 A-1. 의사결정 파이프라인 ────────────────────────────────
def fig_pipeline():
    W, H = 760, 230
    b = [text(380, 26, '학습 설정은 앞 단계가 뒤 단계를 정한다 — 모델 이름보다 정답 y의 모양이 먼저', 14.5, 600)]
    steps = [('문제 정의', GRAY, INK), ('Target 형태', '#FFE7A3', RED), ('출력층', SOFT, PUR), ('Loss', SOFT, PUR),
             ('Optimizer', TSOFT, TEAL), ('LR·Batch·Epoch', TSOFT, TEAL), ('Validation', '#E8F5E9', GRN), ('Metric', '#E8F5E9', GRN)]
    w, gap = 84, 7
    for i, (t, f, c) in enumerate(steps):
        x = 12 + i * (w + gap)
        b.append(box(x, 52, w, 46, fill=f, stroke=c, sw=2 if i == 1 else 1.4))
        if len(t) > 10:   # 긴 이름은 두 줄로
            b.append(text(x + w / 2, 72, 'LR · Batch', 11.5, 700, c)); b.append(text(x + w / 2, 89, '· Epoch', 11.5, 700, c))
        else:
            b.append(text(x + w / 2, 80, t, 12.5, 700, c))
        if i < len(steps) - 1:
            b.append(arrow(x + w, 75, x + w + gap, 75))
    groups = [(1, 3, '정답 y의 모양이 정한다', RED), (4, 5, '학습 방법 — 손실 곡선을 보며 조정', TEAL), (6, 7, '평가 — Loss와 따로 본다', GRN)]
    for a, z, t, c in groups:
        x1, x2 = 12 + a * (w + gap), 12 + z * (w + gap) + w
        b.append(f'<path d="M {x1} 108 L {x1} 116 L {x2} 116 L {x2} 108" fill="none" stroke="{c}" stroke-width="1.4"/>')
        b.append(text((x1 + x2) / 2, 134, t, 11.5, 600, c))
    rows = ['y가 실수 → 회귀 → 출력 1개 → MSELoss / L1Loss',
            'y가 0/1 → 이진분류 → 로짓 1개 → BCEWithLogitsLoss',
            'y가 클래스 번호 → 다중분류 → 로짓 C개 → CrossEntropyLoss']
    for i, r in enumerate(rows):
        b.append(text(380, 166 + i * 22, r, 12.5, 500, INK, mono=False))
    svg('appA_pipeline', W, H, '학습 설정 의사결정 파이프라인',
        '문제 정의, Target 형태, 출력층, Loss, Optimizer, LR·Batch·Epoch, Validation, Metric 순서와 Target에 따른 세 가지 경로', ''.join(b))


# ── 그림 A-2. Target 지도 ───────────────────────────────────────
def fig_target_map():
    W, H = 760, 330
    b = [text(380, 26, 'Target 지도 — 정답 모양 하나로 출력층·Loss·추론·Metric이 함께 정해진다', 14.5, 600)]
    heads = ['정답 y', '문제 유형', '출력 (로짓)', 'Loss', '추론', 'Metric', '이 교재']
    xs = [10, 154, 250, 324, 478, 586, 694]
    ws = [140, 92, 70, 150, 104, 104, 58]
    for x, w, h in zip(xs, ws, heads):
        b.append(box(x, 44, w, 28, fill=GRAY, stroke=LINE, sw=1, rx=3)); b.append(text(x + w / 2, 63, h, 12, 700))
    rows = [('실수 [N, 1] float', '회귀', '1개', 'MSELoss / L1Loss', '그대로', 'MAE·RMSE·R²', '2장', BLU),
            ('0/1 [N, 1] float', '이진분류', '1개', 'BCEWithLogitsLoss', 'sigmoid ≥ 0.5', 'Acc·P·R·F1', '3장', RED),
            ('번호 [N] long', '다중분류', 'C개', 'CrossEntropyLoss', 'argmax', 'Acc·Macro F1', '3–5장', PUR),
            ('0/1 [N, C] float', '다중레이블', 'C개', 'BCEWithLogitsLoss', '칸마다 sigmoid', 'Micro·Macro F1', '부록', TEAL)]
    for r, row in enumerate(rows):
        y = 80 + r * 52
        c = row[-1]
        for k, (x, w) in enumerate(zip(xs, ws)):
            b.append(box(x, y, w, 44, fill='#FFFFFF' if k else SOFT, stroke=c if k in (0, 1, 3) else LINE, sw=1.4 if k in (0, 1, 3) else 1, rx=3))
            b.append(text(x + w / 2, y + 27, row[k], 11.5 if len(row[k]) > 12 else 12, 700 if k in (1, 3) else 500, c if k in (1, 3) else INK, mono=(k == 0)))
    b.append(text(380, 300, '모델 구조(MLP·CNN)는 입력을 처리하는 방식일 뿐 — 같은 다중분류면 출력층과 Loss는 같다', 12.5, 600, RED))
    b.append(text(380, 322, '모델 끝은 로짓을 그대로 내보내고, Sigmoid·Softmax는 Loss 안에서(학습) 또는 추론 때만 쓴다', 12, 400, '#4A4F5A'))
    svg('appA_target_map', W, H, 'Target 지도',
        '회귀, 이진분류, 다중분류, 다중레이블의 정답 모양, 출력 수, Loss, 추론 방법, 평가 지표와 이 교재에서 다룬 장', ''.join(b))


for f in (fig_pipeline, fig_target_map):
    f()
print('생성:', sorted(p.name for p in OUT.glob('appA_*.svg')))
