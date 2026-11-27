"""code/*.py 를 '# %% CODE NN' 단위로 한 이름공간에서 차례로 실행하고 블록별 출력을 JSON으로 저장한다.

- 작업 폴더는 code/ (데이터 파일 housing.csv 등이 여기에 있음)
- matplotlib 그림은 plt.show() 때 figures/<파일>_<NN>_<k>.svg 로 저장하고 JSON에 그림 이름을 적는다.
사용: python tools/run_blocks.py code/<파일>.py   → code/out/<파일>.json
"""
import contextlib, io, json, os, re, sys
from pathlib import Path
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parent.parent
src = Path(sys.argv[1]).resolve(); name = src.stem
out_path = ROOT / 'code/out' / f'{name}.json'
os.chdir(ROOT / 'code')
import matplotlib
matplotlib.rcParams['svg.hashsalt'] = 'dlb'          # 같은 그림이면 같은 SVG가 나오도록
matplotlib.rcParams['svg.fonttype'] = 'none'         # 글자를 글자로 남겨 파일을 작게
import matplotlib.pyplot as plt
state = {'no': None, 'k': 0, 'figs': {}}
for old in (ROOT / 'figures').glob(f'{name}_[0-9][0-9]_*.svg'):
    old.unlink()

def show(*a, **k):
    for num in plt.get_fignums():
        state['k'] += 1
        fname = f"{name}_{state['no']}_{state['k']}"
        plt.figure(num).savefig(ROOT / 'figures' / f'{fname}.svg', bbox_inches='tight', metadata={'Date': None})
        state['figs'].setdefault(state['no'], []).append(fname)
    plt.close('all')
plt.show = show
ns = {'__name__': '__main__'}
blocks = re.split(r'^# %% CODE (\d\d)\n', src.read_text(encoding='utf-8'), flags=re.M)
out = {}
for i in range(1, len(blocks), 2):
    state['no'], state['k'] = blocks[i], 0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(blocks[i + 1], f'{src.name}:CODE{blocks[i]}', 'exec'), ns)
    out[blocks[i]] = buf.getvalue()
out['_figures'] = state['figs']
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(src.name, '블록', len(out) - 1, '그림', sum(len(v) for v in state['figs'].values()))
