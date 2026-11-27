"""실습 PDF의 CODE 블록을 code/<이름>.py 로 옮긴다 (원고 준비용, 한 번 실행).

- PDF 텍스트(pdftotext -layout)에서 'CODE NN | Python 실습 코드' 아래 들여쓴 줄을 코드로 읽는다.
- 같은 이름의 기존 노트북(ml_dl_pytorch)에 공백만 다른 같은 셀이 있으면 빈 줄이 살아 있는 노트북 셀을 쓴다.
- 그 밖에는 PDF 들여쓰기를 쓰고, 문법 검사(compile)에 실패한 블록을 출력한다 → 손으로 고친다.
사용: python tools/extract_lab_code.py <pdf텍스트> <노트북폴더> <이름...>
"""
import json, re, sys
from pathlib import Path
txt = Path(sys.argv[1]).read_text(encoding='utf-8'); nbdir = Path(sys.argv[2]); names = sys.argv[3:]
parts = re.split(r'######## (\S+?)(?:\.pdf)?\s.*\n', txt)
norm = lambda s: re.sub(r'\s+', '', s)
for i in range(1, len(parts), 2):
    name, body = parts[i], parts[i + 1]
    if names and name not in names: continue
    nb = nbdir / f'{name}.ipynb'
    cells = [''.join(c['source']) for c in json.loads(nb.read_text())['cells'] if c['cell_type'] == 'code'] if nb.exists() else []
    bl = re.split(r'\n\s*CODE (\d\d)\s*\| Python 실습 코드\n', body)
    out = [f'# {name}.py — 원본: lab_Pytorch/{name}.pdf 의 CODE 블록을 순서대로 옮김\n']
    for j in range(1, len(bl), 2):
        lines = []
        for l in bl[j + 1].split('\n'):
            if re.match(r'^\s{0,1}\S', l): break
            if re.match(r'^\s+\d+\s*$', l): continue
            lines.append(l)
        ind = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
        # PDF 배치에서 들여쓰기가 1~2칸 어긋나는 일이 있어 4칸 단위로 맞춘다
        code = '\n'.join((' ' * (((len(l) - len(l.lstrip()) - ind) + 1) // 4 * 4) + l.strip()) if l.strip() else ''
                         for l in lines).strip('\n')
        src = 'pdf'
        for c in cells:
            if norm(c) == norm(code): code, src = c.strip('\n'), 'nb'; break
        try: compile(code, 'x', 'exec'); ok = ''
        except SyntaxError as e: ok = f'  ← 문법 오류 {e.lineno}행: {e.msg}'
        print(f'{name} CODE {bl[j]} [{src}]{ok}')
        out.append(f'\n# %% CODE {bl[j]}\n{code}\n')
    Path('code', f'{name}.py').write_text(''.join(out), encoding='utf-8')
