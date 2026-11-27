"""code/*.py 를 '# %% CODE NN' 단위로 한 이름공간에서 차례로 실행하고 블록별 출력을 JSON으로 저장한다."""
import io, json, re, sys, contextlib
from pathlib import Path
src = Path(sys.argv[1]); out = {}
ns = {'__name__': '__main__'}
blocks = re.split(r'^# %% CODE (\d\d)\n', src.read_text(encoding='utf-8'), flags=re.M)
for i in range(1, len(blocks), 2):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(blocks[i+1], f'{src.name}:CODE{blocks[i]}', 'exec'), ns)
    out[blocks[i]] = buf.getvalue()
Path(sys.argv[2]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(src.name, '블록', len(out))
