"""content_src/chNN.md → content/chNN.md

원고의 ⟪code:파일:번호⟫ 줄을 code/파일.py 의 '# %% CODE 번호' 블록과
code/out/파일.json 에 저장된 실제 실행 결과로 바꾼다. 원고에 코드를 손으로 옮겨 적지 않으므로
본문 코드와 실행한 코드가 항상 같다.

실행: python tools/run_blocks.py code/파일.py code/out/파일.json  (실행 결과 갱신)
      python tools/expand_code.py                                  (원고 펼치기)
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = json.loads((ROOT / 'code/out/env.json').read_text(encoding='utf-8'))


def block(name, no):
    src = (ROOT / 'code' / f'{name}.py').read_text(encoding='utf-8')
    parts = re.split(r'^# %% CODE (\d\d)\n', src, flags=re.M)
    table = {parts[i]: parts[i + 1].strip('\n') for i in range(1, len(parts), 2)}
    if no not in table:
        raise KeyError(f'{name} CODE {no} 없음')
    return table[no]


def output(name, no):
    data = json.loads((ROOT / 'code/out' / f'{name}.json').read_text(encoding='utf-8'))
    return data[no].rstrip('\n')


def expand(m):
    name, no = m.group(1), m.group(2)
    code, out = block(name, no), output(name, no)
    s = (f'코드 박스: code/{name}.py · CODE {no} · 실행 확인\n\n```python\n{code}\n```\n')
    if out.strip():
        s += f'\n실행 결과: CODE {no} · {ENV["label"]}\n\n```text\n{out}\n```\n'
    return s


for src in sorted((ROOT / 'content_src').glob('ch*.md')):
    text = re.sub(r'^⟪code:([A-Za-z0-9_]+):(\d\d)⟫$', expand, src.read_text(encoding='utf-8'), flags=re.M)
    (ROOT / 'content' / src.name).write_text(text, encoding='utf-8')
    print('펼침:', src.name, '→ content/' + src.name)
