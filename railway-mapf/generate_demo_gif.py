"""
generate_demo_gif.py
Loads a real flatland test case and renders an animated GIF.
Zero flatland imports - parses raw grid bits with numpy only.

Usage:
    python generate_demo_gif.py

Requirements:
    pip install Pillow numpy

Edit LEVEL / TEST to pick which test case to render.
"""

import os, sys, glob, pickle, math
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageFont

LEVEL = 0
TEST  = 0

OUT_FILE = "demo.gif"
IMG_W    = 860
IMG_H    = 520
PADDING  = 40
FPS      = 18
INTERP   = 6
HOLD_SEC = 2.0

BG           = (5,  10,  14)
RAIL_DOT     = (15, 35,  50)
RAIL_LINE    = (35, 90, 130)
RAIL_CELL    = (22, 60,  88)
TEXT_HI      = (168,216, 234)
TEXT_DIM     = ( 58,106, 130)
GREEN        = (  0,255, 157)
AGENT_COLORS = [
    (  0,255,157),
    (  0,200,255),
    (176,106,255),
    (255,184,  0),
    (255, 60, 90),
    (255,220,100),
]

# Flatland uint16 transition decoder - no flatland import needed
# bits 15-12: facing N, bits 11-8: facing E, bits 7-4: facing S, bits 3-0: facing W
# within each nibble: bit3=N, bit2=E, bit1=S, bit0=W
DELTAS = [(-1,0),(0,1),(1,0),(0,-1)]

def get_trans(grid, r, c, d):
    nibble = (int(grid[r,c]) >> ((3-d)*4)) & 0xF
    return ((nibble>>3)&1, (nibble>>2)&1, (nibble>>1)&1, nibble&1)

def neighbours(grid, r, c, H, W):
    out = set()
    for d in range(4):
        for nd, t in enumerate(get_trans(grid, r, c, d)):
            if not t: continue
            nr, nc = r+DELTAS[nd][0], c+DELTAS[nd][1]
            if 0 <= nr < H and 0 <= nc < W:
                out.add((nr, nc))
    return out

# Load pkl with version-safe unpickler
script_dir = os.path.dirname(os.path.abspath(__file__))
pattern = os.path.join(script_dir, "multi_test_case", f"level{LEVEL}_test_{TEST}.pkl")
matches = glob.glob(pattern)
if not matches:
    print(f"ERROR: no test case at {pattern}")
    sys.exit(1)

print(f"Loading: {matches[0]}")

class _Stub:
    def __init__(self, *a, **kw): pass

class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            return _Stub

with open(matches[0], "rb") as f:
    data = CompatUnpickler(f).load()

def _get(obj, *keys):
    d = obj if isinstance(obj, dict) else getattr(obj, "__dict__", {})
    for k in keys:
        if k in d: return d[k]
        if hasattr(obj, k): return getattr(obj, k)
    return None

raw_grid     = _get(data, "grid")
agents_raw   = _get(data, "agents")
max_timestep = _get(data, "max_episode_steps", "max_timestep") or 200

if raw_grid is None or agents_raw is None:
    print("ERROR: could not read pkl fields. Contents:")
    d = data if isinstance(data, dict) else getattr(data, "__dict__", {})
    for k,v in d.items(): print(f"  {k}: {type(v)}")
    sys.exit(1)

grid = np.array(raw_grid, dtype=np.uint16)
H, W = grid.shape
print(f"Grid: {H}x{W}  |  Agents: {len(agents_raw)}  |  Max steps: {max_timestep}")

class Agent:
    pass

agents = []
for ag in agents_raw:
    d  = getattr(ag, "__dict__", {})
    ip = d.get("initial_position") or getattr(ag, "initial_position", None)
    tg = d.get("target")           or getattr(ag, "target", None)
    if ip is not None and tg is not None:
        a = Agent()
        a.initial_position = ip
        a.target = tg
        agents.append(a)

print(f"Valid agents: {len(agents)}")

# BFS planner
def bfs(start, goal, blocked=set()):
    if start == goal: return [start]
    q, vis = deque([(start,[start])]), {start}
    while q:
        cur, path = q.popleft()
        for nb in neighbours(grid, *cur, H, W):
            if nb in vis or nb in blocked: continue
            vis.add(nb)
            if nb == goal: return path + [nb]
            q.append((nb, path+[nb]))
    return [start]

print("Planning paths...")
paths, occupied = [], set()
for i, ag in enumerate(agents):
    p = bfs(ag.initial_position, ag.target, occupied)
    paths.append(p)
    occupied.update(p)
    print(f"  Agent {i+1}: {ag.initial_position} -> {ag.target}  len={len(p)}")

rail_cells = {(r,c) for r in range(H) for c in range(W) if grid[r,c] != 0}

cell = max(4, min((IMG_W-2*PADDING)/W, (IMG_H-2*PADDING-50)/H))
ox   = (IMG_W - cell*W) / 2
oy   = (IMG_H - cell*H) / 2 + 18

def gx(c): return ox + (c+0.5)*cell
def gy(r): return oy + (r+0.5)*cell

def load_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
               "/System/Library/Fonts/Menlo.ttc",
               "C:/Windows/Fonts/consola.ttf"]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()

fnt_title = load_font(14)
fnt_small = load_font(10)

def draw_grid(draw):
    s = max(1, cell*0.07)
    for r in range(H):
        for c in range(W):
            x,y = gx(c),gy(r)
            draw.ellipse([x-s,y-s,x+s,y+s], fill=RAIL_DOT)

def draw_rails(draw):
    drawn = set()
    for (r,c) in rail_cells:
        for (nr,nc) in neighbours(grid, r, c, H, W):
            key = (min(r,nr),min(c,nc),max(r,nr),max(c,nc))
            if key in drawn: continue
            drawn.add(key)
            draw.line([(gx(c),gy(r)),(gx(nc),gy(nr))], fill=RAIL_LINE, width=max(2,int(cell*0.22)))
    for (r,c) in rail_cells:
        s = max(2,cell*0.11); x,y = gx(c),gy(r)
        draw.ellipse([x-s,y-s,x+s,y+s], fill=RAIL_CELL)

def draw_goals(draw, pulse):
    for i,ag in enumerate(agents):
        col = AGENT_COLORS[i%len(AGENT_COLORS)]
        r0,c0 = ag.target; x,y = gx(c0),gy(r0)
        rv = cell*0.26+pulse*cell*0.05
        draw.ellipse([x-rv,y-rv,x+rv,y+rv], outline=tuple(int(v*0.45) for v in col), width=max(1,int(cell*0.07)))
        s = cell*0.14
        draw.polygon([(x,y-s),(x+s,y),(x,y+s),(x-s,y)], outline=col)

def draw_trail(draw, path, color, t):
    if t < 1: return
    dim = tuple(int(v*0.28) for v in color)
    lw  = max(2, int(cell*0.16))
    for i in range(1, min(t+1,len(path))):
        r1,c1=path[i-1]; r2,c2=path[i]
        draw.line([(gx(c1),gy(r1)),(gx(c2),gy(r2))], fill=dim, width=lw)

def draw_agent(draw, img, path, color, t, sub):
    ti = min(t, len(path)-1)
    if ti < len(path)-1:
        r1,c1=path[ti]; r2,c2=path[ti+1]
        x=gx(c1)+(gx(c2)-gx(c1))*sub; y=gy(r1)+(gy(r2)-gy(r1))*sub
    else:
        r,c=path[-1]; x,y=gx(c),gy(r)
    R = max(4, int(cell*0.28))
    gs = R*4
    gi = Image.new("RGBA",(gs*2,gs*2),(0,0,0,0))
    ImageDraw.Draw(gi).ellipse([0,0,gs*2-1,gs*2-1], fill=color+(45,))
    img.alpha_composite(gi, (int(x)-gs,int(y)-gs))
    draw.ellipse([x-R,y-R,x+R,y+R], fill=color)
    ir = max(2,R//3)
    draw.ellipse([x-ir,y-ir,x+ir,y+ir], fill=BG)

def draw_hud(draw, t, total, n, arrived):
    draw.rectangle([0,0,IMG_W,26], fill=(7,18,28))
    draw.text((12,6), "RAILWAY MAPF  -  Multi-Agent Path Finding", font=fnt_title, fill=GREEN)
    info = f"t={t:03d}/{total:03d}  agents={n}  arrived={arrived}/{n}"
    try:   tw = draw.textbbox((0,0),info,font=fnt_small)[2]
    except: tw = len(info)*6
    draw.text((IMG_W-tw-12,8), info, font=fnt_small, fill=TEXT_DIM)
    for i in range(min(len(agents),6)):
        col=AGENT_COLORS[i%len(AGENT_COLORS)]; lx,ly=12+i*100,IMG_H-20
        if lx+90>IMG_W: break
        draw.ellipse([lx,ly-4,lx+8,ly+4], fill=col)
        draw.text((lx+12,ly-5), f"Agent {i+1}", font=fnt_small, fill=TEXT_HI)

max_t  = max(len(p) for p in paths)
hold_n = int(HOLD_SEC*FPS)
print(f"Rendering {max_t} steps x {INTERP} sub-frames...")
frames = []

for t in range(max_t):
    for si in range(INTERP):
        sub   = si/INTERP
        pulse = 0.5+0.5*math.sin((t*INTERP+si)*0.28)
        img   = Image.new("RGBA",(IMG_W,IMG_H),BG)
        draw  = ImageDraw.Draw(img,"RGBA")
        draw_grid(draw); draw_rails(draw); draw_goals(draw,pulse)
        arrived = sum(1 for p in paths if t>=len(p)-1)
        for i,p in enumerate(paths): draw_trail(draw,p,AGENT_COLORS[i%len(AGENT_COLORS)],t)
        for i,p in enumerate(paths): draw_agent(draw,img,p,AGENT_COLORS[i%len(AGENT_COLORS)],t,sub)
        draw_hud(draw,t,max_t-1,len(agents),arrived)
        frames.append(img.convert("P",palette=Image.ADAPTIVE,dither=0))

for _ in range(hold_n):
    frames.append(frames[-1])

print(f"Saving {OUT_FILE} ({len(frames)} frames)...")
frames[0].save(OUT_FILE,save_all=True,append_images=frames[1:],
               optimize=False,duration=int(1000/FPS),loop=0)
print(f"Done! {os.path.getsize(OUT_FILE)//1024} KB -> {OUT_FILE}")
