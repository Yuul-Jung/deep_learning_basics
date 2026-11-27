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


