"""SVG 그림 공통 도구 — 색·글꼴·상자·글자·화살표·칸 표."""
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

def plot_frame(x0, y0, w, h, xr, yr, xt, yt, xl='', yl=''):
    """좌표축과 눈금. 반환: (그림 조각, 좌표 변환 함수)"""
    X = lambda v: x0 + (v - xr[0]) / (xr[1] - xr[0]) * w
    Y = lambda v: y0 + h - (v - yr[0]) / (yr[1] - yr[0]) * h
    b = [f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#FBFBF9" stroke="{LINE}" stroke-width="1"/>']
    for v in xt:
        b.append(f'<line x1="{X(v):.1f}" y1="{y0}" x2="{X(v):.1f}" y2="{y0 + h}" stroke="#E6E7E2" stroke-width="1"/>')
        b.append(text(X(v), y0 + h + 16, f'{v:g}', 11, 400, '#4A4F5A'))
    for v in yt:
        b.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x0 + w}" y2="{Y(v):.1f}" stroke="#E6E7E2" stroke-width="1"/>')
        b.append(text(x0 - 6, Y(v) + 4, f'{v:g}', 11, 400, '#4A4F5A', 'end'))
    if xl: b.append(text(x0 + w / 2, y0 + h + 34, xl, 12, 500))
    if yl: b.append(f'<text x="{x0 - 34}" y="{y0 + h / 2}" font-size="12" font-weight="500" fill="{INK}" text-anchor="middle" transform="rotate(-90 {x0 - 34} {y0 + h / 2})">{escape(yl)}</text>')
    return ''.join(b), X, Y


def poly(points, color, sw=2, dash=False):
    d = ' stroke-dasharray="6 4"' if dash else ''
    pts = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{sw}"{d}/>'


def dot(x, y, color, r=5, fill=None):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill or color}" stroke="{color}" stroke-width="1.5"/>'
