from __future__ import annotations

import itertools
import math
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache

MAX_DIGITS = 2000


class RuleError(ValueError):
    pass


CATALOG: dict[str, dict] = {
    "A": dict(title="순서 정렬", modes=["asc", "desc", "high", "low"], default="high"),
    "B": dict(title="순환 이동", modes=["left", "right"], default="left"),
    "C": dict(title="전체 반전", modes=["reverse"], default="reverse"),
    "D": dict(title="부분 반전", modes=["odd", "even"], default="odd"),
    "E": dict(title="셔플", modes=["shuffle", "inverse"], default="shuffle"),
    "I": dict(title="상수 더하기", modes=["add"], default="add"),
    "J": dict(title="상수 곱하기", modes=["mul"], default="mul"),
    "K": dict(title="거듭제곱", modes=["square", "cube", "pow"], default="square"),
    "M": dict(title="중간 모듈러", modes=["mod"], default="mod"),
    "P": dict(title="자릿수 변환", modes=["reverse", "sum", "product"], default="reverse"),
    "Q": dict(title="기준값 저장", modes=["first", "last", "min", "max", "kth"], default="first"),
    "R": dict(title="저장값과 결합", modes=["add", "mul"], default="add"),
    "S": dict(title="전체 통계 더하기", modes=["sum", "min", "max"], default="sum"),
    "AF": dict(title="이웃과 계산", modes=["sum", "product", "diff"], default="diff"),
    "AG": dict(title="두 원소씩 교환", modes=["swap"], default="swap"),
    "X": dict(title="위치와 결합", modes=["add", "mul", "pow"], default="add"),
    "Y": dict(title="홀짝 위치 연산", modes=["add", "mul"], default="add"),
    "Z": dict(title="홀짝 값 조건", modes=["even", "odd"], default="even"),
    "AB": dict(title="비교 조건", modes=["above", "below", "previous"], default="above"),
    "AJ": dict(title="대상 선택",
               modes=["all", "oddIndex", "evenIndex", "oddValue", "evenValue", "threshold"],
               default="oddIndex"),
    "AO": dict(title="주기적 상수", modes=["add", "mul"], default="add"),
    "BA": dict(title="소인수 변환", modes=["largest", "smallest", "omega", "distinct"],
               default="largest"),
    "BC": dict(title="GCD · LCM 결합", modes=["addG", "mulG", "addL", "mulL"], default="addG"),
    "REF": dict(title="참조 배선", modes=["sortnext", "sum", "first", "next", "perm", "map"],
                default="first"),
}
RULE_LAB_CODES = [c for c in CATALOG if c != "REF"]
FN_KEYS = ["keep", "add", "mul", "pow", "mod", "reverse", "sum", "product"]
DEFAULT_PARAMS = dict(c="2", k=None, a="2", b="3", t="10", period="2|5|3", fn="add", fc="2",
                      op="add", seed="0", tseed="0")


@dataclass(frozen=True)
class Step:
    code: str
    mode: str
    params: tuple = ()

    def p(self, key):
        d = dict(self.params)
        if key in d:
            return d[key]
        if key == "k":
            return "1" if self.code == "B" else "2"
        return DEFAULT_PARAMS[key]

    def __str__(self):
        s = f"{self.code}:{self.mode}"
        if self.params:
            s += "(" + ",".join(f"{k}={v}" for k, v in self.params) + ")"
        return s


_STEP_RE = re.compile(r"^\s*([A-Z]+)\s*(?::\s*([A-Za-z]+))?\s*(?:\((.*)\))?\s*$")


def parse_step(text: str) -> Step:
    m = _STEP_RE.match(text)
    if not m:
        raise RuleError(f"단계를 읽을 수 없습니다: {text!r}")
    code, mode, args = m.group(1), m.group(2), m.group(3)
    if code not in CATALOG:
        raise RuleError(f"없는 규칙 코드입니다: {code}")
    mode = mode or CATALOG[code]["default"]
    if mode not in CATALOG[code]["modes"]:
        raise RuleError(f"{code}의 모드가 아님: {mode} (가능: {CATALOG[code]['modes']})")
    params = {}
    if args and args.strip():
        for part in args.split(","):
            if "=" not in part:
                raise RuleError(f"매개변수는 key=value 꼴이어야 함: {part!r}")
            k, v = (s.strip() for s in part.split("=", 1))
            if k not in DEFAULT_PARAMS:
                raise RuleError(f"없는 매개변수입니다: {k}")
            params[k] = v
    return Step(code, mode, tuple(sorted(params.items())))


def parse_spec(spec: str) -> list[Step]:
    spec = PRESETS.get(spec, spec)
    if not spec.strip():
        return []
    return [parse_step(s) for s in spec.split(">")]


def spec_str(steps: list[Step]) -> str:
    return " > ".join(str(s) for s in steps)


def _chk(x: int) -> int:
    if x.bit_length() > MAX_DIGITS * 3.33:
        raise RuleError("중간값이 너무 큼")
    return x


def _pos_int(v, label, zero=False) -> int:
    s = str(v).strip()
    if not s.isdigit():
        raise RuleError(f"{label}은 정수여야 함: {v!r}")
    x = int(s)
    if x < (0 if zero else 1):
        raise RuleError(f"{label} 범위 오류: {x}")
    return x


def _pow(x: int, e: int) -> int:
    if x == 1 or e == 1:
        return x
    if (x.bit_length() - 1) * e > MAX_DIGITS * 3.33:
        raise RuleError("거듭제곱 결과가 너무 큼")
    return x ** e


def _digits(x: int, mode: str) -> int:
    s = str(x)
    if mode == "reverse":
        return int(s[::-1])
    if mode == "sum":
        return sum(int(d) for d in s)
    out = 1
    for d in s:
        if d != "0":
            out *= int(d)
    return out


@lru_cache(maxsize=1 << 16)
def _factors(x: int) -> dict:
    if x == 1:
        return dict(smallest=1, largest=1, omega=1, distinct=1)
    rest, d, first, last, count, distinct, ticks = x, 2, 0, 1, 0, 0, 0
    while d * d <= rest:
        ticks += 1
        if ticks > 300000:
            raise RuleError("소인수 계산이 너무 큽니다")
        if rest % d == 0:
            distinct += 1
            if not first:
                first = d
            last = d
            while rest % d == 0:
                rest //= d
                count += 1
        d = 3 if d == 2 else d + 2
    if rest > 1:
        if not first:
            first = rest
        last = rest
        count += 1
        distinct += 1
    return dict(smallest=first, largest=last, omega=count + 1, distinct=distinct + 1)


def _fn_for(step: Step, n: int):
    fn = step.p("fn")
    if fn not in FN_KEYS:
        raise RuleError(f"변환 이름 오류: {fn}")
    c = _pos_int(step.p("fc"), "변환 상수") if fn in ("add", "mul", "pow") else 1

    def f(x):
        if fn == "keep":
            return x
        if fn == "add":
            return _chk(x + c)
        if fn == "mul":
            return _chk(x * c)
        if fn == "pow":
            return _pow(x, c)
        if fn == "mod":
            return x % n + 1
        return _digits(x, fn)
    return f


@lru_cache(maxsize=256)
def _wiring(kind: str, m: int, seed: int) -> tuple:
    rng = random.Random(f"wiring-{kind}-{m}-{seed}")
    if kind == "perm":
        while True:
            p = list(range(m))
            rng.shuffle(p)
            if all(p[i] != i for i in range(m)):
                return tuple(p)
    return tuple(rng.choice([j for j in range(m) if j != i]) for i in range(m))


@lru_cache(maxsize=64)
def _op_table(n: int, seed: int) -> tuple:
    rng = random.Random(f"optable-{n}-{seed}")
    return tuple(tuple(rng.randint(1, n) for _ in range(n)) for _ in range(n))


def apply_step(xs: list[int], step: Step, n: int, r):
    code, mode, m = step.code, step.mode, len(xs)
    ys = list(xs)
    if code == "A":
        s = sorted(xs)
        if mode == "asc":
            ys = s
        elif mode == "desc":
            ys = s[::-1]
        else:
            ys, lo, hi, take_high = [], 0, m - 1, mode == "high"
            while lo <= hi:
                if take_high:
                    ys.append(s[hi]); hi -= 1
                else:
                    ys.append(s[lo]); lo += 1
                take_high = not take_high
    elif code == "B":
        k = _pos_int(step.p("k"), "이동량 k", zero=True) % m
        shift = k if mode == "left" else (m - k) % m
        ys = xs[shift:] + xs[:shift]
    elif code == "C":
        ys = xs[::-1]
    elif code == "D":
        idx = [i for i in range(m) if i % 2 == (0 if mode == "odd" else 1)]
        vals = [xs[i] for i in idx][::-1]
        for i, v in zip(idx, vals):
            ys[i] = v
    elif code == "E":
        if mode == "inverse":
            ys = xs[0::2] + xs[1::2]
        else:
            half = math.ceil(m / 2)
            ys = []
            for i in range(half):
                ys.append(xs[i])
                if i + half < m:
                    ys.append(xs[i + half])
    elif code == "I":
        c = _pos_int(step.p("c"), "상수 c")
        ys = [_chk(x + c) for x in xs]
    elif code == "J":
        c = _pos_int(step.p("c"), "상수 c")
        ys = [_chk(x * c) for x in xs]
    elif code == "K":
        e = 2 if mode == "square" else 3 if mode == "cube" else _pos_int(step.p("k"), "지수 k")
        ys = [_pow(x, e) for x in xs]
    elif code == "M":
        ys = [x % n + 1 for x in xs]
    elif code == "P":
        ys = [_digits(x, mode) for x in xs]
    elif code == "Q":
        if mode == "first":
            r = xs[0]
        elif mode == "last":
            r = xs[-1]
        elif mode == "min":
            r = min(xs)
        elif mode == "max":
            r = max(xs)
        else:
            k = _pos_int(step.p("k"), "선택 위치 k")
            if k > m:
                raise RuleError("선택 위치 k가 배열 길이보다 큼")
            r = xs[k - 1]
    elif code == "R":
        if r is None:
            raise RuleError("R 앞에 Q가 필요")
        ys = [_chk(x + r if mode == "add" else x * r) for x in xs]
    elif code == "S":
        v = sum(xs) if mode == "sum" else min(xs) if mode == "min" else max(xs)
        ys = [_chk(x + v) for x in xs]
    elif code == "AF":
        out = []
        for i, x in enumerate(xs):
            y = xs[(i + 1) % m]
            out.append(_chk(x + y if mode == "sum" else x * y if mode == "product" else (y - x) ** 2 + 1))
        ys = out
    elif code == "AG":
        for i in range(0, m - 1, 2):
            ys[i], ys[i + 1] = ys[i + 1], ys[i]
    elif code == "X":
        if mode == "pow":
            ys = [_pow(x, i + 1) for i, x in enumerate(xs)]
        else:
            ys = [_chk(x + (i + 1) if mode == "add" else x * (i + 1)) for i, x in enumerate(xs)]
    elif code == "Y":
        a, b = _pos_int(step.p("a"), "a"), _pos_int(step.p("b"), "b")
        ys = [_chk((x + (a if i % 2 == 0 else b)) if mode == "add" else x * (a if i % 2 == 0 else b))
              for i, x in enumerate(xs)]
    elif code in ("Z", "AB", "AJ"):
        f = _fn_for(step, n)
        total = sum(xs) if code == "AB" else 0
        t = _pos_int(step.p("t"), "기준값 t") if (code == "AJ" and mode == "threshold") else 1
        mask = []
        for i, x in enumerate(xs):
            if code == "Z":
                mask.append(x % 2 == (0 if mode == "even" else 1))
            elif code == "AB":
                if mode == "above":
                    mask.append(x * m > total)
                elif mode == "below":
                    mask.append(x * m < total)
                else:
                    mask.append(x > xs[(i + m - 1) % m])
            else:
                mask.append({"all": True, "oddIndex": i % 2 == 0, "evenIndex": i % 2 == 1,
                             "oddValue": x % 2 == 1, "evenValue": x % 2 == 0,
                             "threshold": x >= t}[mode])
        ys = [f(x) if mk else x for x, mk in zip(xs, mask)]
    elif code == "AO":
        vals = [_pos_int(v, "주기 상수") for v in re.split(r"[|;\s]+", str(step.p("period"))) if v]
        if not vals:
            raise RuleError("주기 상수가 비었습니다")
        ys = [_chk(x + vals[i % len(vals)] if mode == "add" else x * vals[i % len(vals)])
              for i, x in enumerate(xs)]
    elif code == "BA":
        ys = [_factors(x)[mode] for x in xs]
    elif code == "BC":
        if mode.endswith("G"):
            v = 0
            for x in xs:
                v = math.gcd(v, x)
        else:
            v = 1
            for x in xs:
                v = _chk(v // math.gcd(v, x) * x)
        ys = [_chk(x + v if mode.startswith("add") else x * v) for x in xs]
    elif code == "REF":
        ys = _apply_ref(xs, step, n)
    else:
        raise RuleError(code)
    if any(y < 1 for y in ys):
        raise RuleError("양의 정수 조건을 벗어났습니다")
    return ys, r


COMMUTATIVE_OPS = ("add", "mul")


def _apply_ref(xs, step: Step, n: int):
    m, mode = len(xs), step.mode
    op = step.p("op")
    if op not in ("add", "add3", "mul", "table"):
        raise RuleError("REF의 op는 add, add3, mul, table 중 하나")
    if op == "table":
        tab = _op_table(n, int(step.p("tseed")))
        T = lambda a, b: tab[(a - 1) % n][(b - 1) % n]
    elif op == "mul":
        T = lambda a, b: a * b
    elif op == "add3":
        T = lambda a, b: a + 3 * b
    else:
        T = lambda a, b: a + b
    if mode == "sortnext":
        s = sorted(xs)
        return [T(s[i], s[(i + 1) % m]) for i in range(m)]
    if mode == "sum":
        tot = sum(xs)
        return [T(x, tot) for x in xs]
    if mode == "first":
        return [T(x, xs[0]) for x in xs]
    if mode == "next":
        return [T(xs[i], xs[(i + 1) % m]) for i in range(m)]
    w = _wiring(mode, m, int(step.p("seed")))
    return [T(xs[i], xs[w[i]]) for i in range(m)]


def ref_pairs(step: Step, xs, n: int):
    m, mode = len(xs), step.mode
    red = lambda v: (v - 1) % n + 1
    if mode == "sortnext":
        s = sorted(xs)
        return [(s[i], s[(i + 1) % m]) for i in range(m)]
    if mode == "sum":
        t = red(sum(xs))
        return [(x, t) for x in xs]
    if mode == "first":
        return [(x, xs[0]) for x in xs]
    if mode == "next":
        return [(xs[i], xs[(i + 1) % m]) for i in range(m)]
    w = _wiring(mode, m, int(step.p("seed")))
    return [(xs[i], xs[w[i]]) for i in range(m)]


def ref_graph(rule: "Rule", m: int = 5) -> dict | None:
    if len(rule.steps) != 1 or rule.steps[0].code != "REF":
        return None
    st = rule.steps[0]
    probe = list(range(1, m + 1))
    pairs = ref_pairs(st, probe, rule.n) if st.mode not in ("sum", "sortnext") else []
    if not pairs:
        return dict(edges_unordered=None, edges_ordered=None, connected=None)
    und = {frozenset(p) for p in pairs}
    parent = list(range(m + 1))

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for e in und:
        if len(e) == 2:
            a, b = tuple(e)
            parent[find(a)] = find(b)
    connected = len({find(v) for v in probe}) == 1
    return dict(edges_unordered=len(und), edges_ordered=len(set(pairs)), connected=connected)


def op_coverage(rule: "Rule", train_x) -> float | None:
    if len(rule.steps) != 1 or rule.steps[0].code != "REF":
        return None
    st = rule.steps[0]
    commutative = st.p("op") in COMMUTATIVE_OPS
    seen = set()
    for row in train_x:
        for a, b in ref_pairs(st, list(row), rule.n):
            seen.add(frozenset((a, b)) if commutative else (a, b))
    n = rule.n
    return len(seen) / (n * (n + 1) / 2 if commutative else n * n)


@dataclass
class Rule:
    spec: str
    n: int
    steps: list = field(default_factory=list)

    def __post_init__(self):
        self.steps = parse_spec(self.spec)
        self.spec = spec_str(self.steps)
        saved = False
        for s in self.steps:
            if s.code == "Q":
                saved = True
            if s.code == "R" and not saved:
                raise RuleError("R 앞에 Q가 필요합니다")

    def trace(self, xs):
        cur, r, out = list(xs), None, []
        for s in self.steps:
            cur, r = apply_step(cur, s, self.n, r)
            out.append(cur)
        return out

    def raw(self, xs):
        cur, r = list(xs), None
        for s in self.steps:
            cur, r = apply_step(cur, s, self.n, r)
        return cur

    def __call__(self, xs) -> tuple:
        return tuple(x % self.n + 1 for x in self.raw(xs))


PRESETS: dict[str, str] = {
    "sort": "A:asc",
    "add_c": "I:add(c=3)",
    "add_total": "S:sum",
    "add_first": "Q:first > R:add",
    "nbsum": "AF:sum",
    "add_pos": "X:add",
    "l1_sortnext": "REF:sortnext(op=add)",
    "l1_sum": "REF:sum(op=add)",
    "l1_first": "REF:first(op=add)",
    "l1_next": "REF:next(op=add)",
    "l1_perm": "REF:perm(op=add,seed=0)",
    "l1_map": "REF:map(op=add,seed=0)",
    "l1t_sortnext": "REF:sortnext(op=table)",
    "l1t_sum": "REF:sum(op=table)",
    "l1t_first": "REF:first(op=table)",
    "l1t_next": "REF:next(op=table)",
    "l1t_perm": "REF:perm(op=table,seed=0)",
    "l1t_map": "REF:map(op=table,seed=0)",
    "w5_next": "REF:next(op=add)",
    "w5_p0": "REF:perm(op=add,seed=0)",
    "w5_p2": "REF:perm(op=add,seed=2)",
    "w6_p1": "REF:perm(op=add,seed=1)",
    "w6_p4": "REF:perm(op=add,seed=4)",
    "w6_p6": "REF:perm(op=add,seed=6)",
    "w2_m4": "REF:map(op=add,seed=4)",
    "w2_m12": "REF:map(op=add,seed=12)",
    "w2_m15": "REF:map(op=add,seed=15)",
    "w1_m0": "REF:map(op=add,seed=0)",
    "w1_m2": "REF:map(op=add,seed=2)",
    "w1_m8": "REF:map(op=add,seed=8)",
    "w5_next_n3": "REF:next(op=add3)",
    "w5_next_mul": "REF:next(op=mul)",
    "w5_p2_n3": "REF:perm(op=add3,seed=2)",
    "w6_p1_n3": "REF:perm(op=add3,seed=1)",
    "w6_p4_n3": "REF:perm(op=add3,seed=4)",
    "w2_m4_n3": "REF:map(op=add3,seed=4)",
    "w2_m12_n3": "REF:map(op=add3,seed=12)",
    "w2_m15_n3": "REF:map(op=add3,seed=15)",
    "w1_m0_n3": "REF:map(op=add3,seed=0)",
    "p_sort": "A:asc",
    "p_next": "REF:next(op=add)",
    "p_first": "Q:first > R:add",
    "comp_sort_next": "A:asc > REF:next(op=add)",
    "comp_next_next": "REF:next(op=add) > REF:next(op=add)",
    "comp_first_sort": "Q:first > R:add > A:asc",
    "comp_next_sort": "REF:next(op=add) > A:asc",
    "c_sum_sort_nb": "S:sum > A:asc > AF:sum",
    "c_sort_nb_sort": "A:asc > AF:sum > A:asc",
    "c_nb_sort_pos": "AF:sum > A:asc > X:add",
    "c_pos_sort_sum": "X:add > A:asc > S:sum",
    "ctrl_add_mul_sort": "I:add(c=3) > J:mul(c=3) > A:asc",
    "ctrl_sort_rot_rev": "A:asc > B:left(k=1) > C:reverse",
}


def random_inputs(n: int, m: int, count: int, seed: int = 0) -> list[tuple]:
    rng = random.Random(seed)
    return [tuple(rng.sample(range(1, n + 1), m)) for _ in range(count)]


def permute(xs, sigma):
    return tuple(xs[j] for j in sigma)


def symmetry(rule, n, m, probes):
    perms = list(itertools.permutations(range(m)))
    base = [rule(x) for x in probes]
    inv, eq, permuting = [], [], []
    short = probes[: min(len(probes), 64)]
    for sigma in perms:
        outs = [rule(permute(x, sigma)) for x in probes]
        if all(o == b for o, b in zip(outs, base)):
            inv.append(sigma)
        if all(o == permute(b, sigma) for o, b in zip(outs, base)):
            eq.append(sigma)
        cands = None
        for x, o, b in zip(short, outs, base):
            ok = {t for t in perms if permute(b, t) == o} if cands is None else \
                 {t for t in cands if permute(b, t) == o}
            cands = ok
            if not cands:
                break
        if cands:
            permuting.append(sigma)
    return dict(invariant=inv, equivariant=eq, permuting=permuting)


def dependency(rule, n, m, probes, seed=0):
    rng = random.Random(seed)
    infl = [[0] * m for _ in range(m)]
    for x in probes:
        base = rule(x)
        unused = [v for v in range(1, n + 1) if v not in x]
        for j in range(m):
            y = list(x)
            y[j] = rng.choice(unused)
            out = rule(tuple(y))
            for i in range(m):
                if out[i] != base[i]:
                    infl[i][j] += 1
    infl = [[c / len(probes) for c in row] for row in infl]
    per_out = [sum(1 for v in row if v > 0) for row in infl]
    per_in = [sum(1 for i in range(m) if infl[i][j] > 0) for j in range(m)]
    return dict(influence=infl, per_output=per_out, per_input=per_in,
                mean=sum(per_out) / m, max=max(per_out))


def classify_step(before: list[tuple], after: list[tuple]) -> str:
    if all(tuple(a) == tuple(b) for a, b in zip(before, after)):
        return "identity"
    m = len(before[0])
    for tau in itertools.permutations(range(m)):
        if all(tuple(a) == permute(b, tau) for b, a in zip(before, after)):
            return "fixed_perm"
    g, ok = {}, True
    for b, a in zip(before, after):
        for x, y in zip(b, a):
            if g.setdefault(x, y) != y:
                ok = False
                break
        if not ok:
            break
    if ok:
        return "relabel" if len(set(g.values())) == len(g) else "elementwise_lossy"
    return "core"


def effective_depth(rule: Rule, probes) -> dict:
    states = [list(map(tuple, (list(p) for p in probes)))]
    traces = [rule.trace(p) for p in probes]
    for k in range(len(rule.steps)):
        states.append([tuple(t[k]) for t in traces])
    kinds = [classify_step(states[k], states[k + 1]) for k in range(len(rule.steps))]
    depth = sum(1 for k in kinds if k in ("core", "elementwise_lossy"))
    return dict(step_kinds=kinds, effective_depth=depth, nominal_depth=len(rule.steps))


def group_label(size: int, m: int) -> str:
    if size == math.factorial(m):
        return f"S{m}"
    if size == math.factorial(m - 1):
        return f"S{m-1}?"
    if size == m:
        return f"Z{m}?"
    if size == 1:
        return "trivial"
    return f"order {size}"


def analyze(spec: str, n: int = 50, m: int = 5, n_probe: int = 200, seed: int = 0) -> dict:
    rule = Rule(spec, n)
    probes = random_inputs(n, m, n_probe, seed)
    outs = [rule(p) for p in probes]
    sym = symmetry(rule, n, m, probes)
    dep = dependency(rule, n, m, probes, seed)
    depth = effective_depth(rule, probes)
    constant = len(set(outs)) == 1
    n_inv, n_eq, n_perm = len(sym["invariant"]), len(sym["equivariant"]), len(sym["permuting"])
    if n_inv > 1 and n_inv >= n_eq:
        kind = "invariant"
    elif n_eq > 1:
        kind = "equivariant"
    elif n_perm > 1:
        kind = "permuting"
    else:
        kind = "none"
    return dict(
        spec=rule.spec, n=n, m=m,
        sym_kind=kind,
        inv_size=n_inv, eq_size=n_eq, permuting_size=n_perm,
        inv_label=group_label(n_inv, m), eq_label=group_label(n_eq, m),
        inv_group=[list(s) for s in sym["invariant"]],
        eq_group=[list(s) for s in sym["equivariant"]],
        dep_mean=dep["mean"], dep_max=dep["max"], dep_per_output=dep["per_output"],
        influence=dep["influence"],
        **depth,
        constant=constant,
        trivial=constant or depth["effective_depth"] == 0,
        pointwise=(n_eq == math.factorial(m) and dep["max"] <= 1),
        ref_graph=ref_graph(rule, m),
        distinct_outputs_frac=len(set(outs)) / len(outs),
    )


_SAMPLE_PARAMS = dict(c=["2", "3", "5", "7"], k=["1", "2", "3"], a=["2", "3"], b=["3", "5"],
                      t=["10", "25"], fc=["2", "3"], period=["2|5|3", "1|4"])


def sample_step(rng: random.Random, code: str | None = None) -> Step:
    code = code or rng.choice(RULE_LAB_CODES)
    mode = rng.choice(CATALOG[code]["modes"])
    params = {}
    if code in ("I", "J"):
        params["c"] = rng.choice(_SAMPLE_PARAMS["c"])
    if code == "B":
        params["k"] = rng.choice(["1", "2"])
    if code == "K" and mode == "pow":
        params["k"] = rng.choice(["2", "3", "4"])
    if code == "Q" and mode == "kth":
        params["k"] = rng.choice(["2", "3", "4"])
    if code == "Y":
        params["a"], params["b"] = rng.choice(_SAMPLE_PARAMS["a"]), rng.choice(_SAMPLE_PARAMS["b"])
    if code == "AO":
        params["period"] = rng.choice(_SAMPLE_PARAMS["period"])
    if code == "AJ" and mode == "threshold":
        params["t"] = rng.choice(_SAMPLE_PARAMS["t"])
    if code in ("Z", "AB", "AJ"):
        params["fn"] = rng.choice([f for f in FN_KEYS if f != "keep"])
        if params["fn"] in ("add", "mul", "pow"):
            params["fc"] = rng.choice(_SAMPLE_PARAMS["fc"])
    return Step(code, mode, tuple(sorted(params.items())))


def sample_rules(count: int, n: int = 50, m: int = 5, min_steps: int = 1, max_steps: int = 4,
                 seed: int = 0, n_probe: int = 100, keep_trivial: bool = False,
                 keep_pointwise: bool = False, max_tries: int = 100000) -> list[dict]:
    rng = random.Random(seed)
    probes = random_inputs(n, m, n_probe, seed + 1)
    seen_outputs, out, tries = set(), [], 0
    while len(out) < count and tries < max_tries:
        tries += 1
        steps, saved = [], False
        for _ in range(rng.randint(min_steps, max_steps)):
            s = sample_step(rng)
            if s.code == "R" and not saved:
                steps.append(sample_step(rng, "Q"))
                saved = True
            if s.code == "Q":
                saved = True
            steps.append(s)
        spec = spec_str(steps)
        try:
            rule = Rule(spec, n)
            sig = tuple(rule(p) for p in probes)
        except RuleError:
            continue
        if sig in seen_outputs:
            continue
        try:
            info = analyze(spec, n, m, n_probe=n_probe, seed=seed + 2)
        except RuleError:
            continue
        if info["trivial"] and not keep_trivial:
            continue
        if info["pointwise"] and not keep_pointwise:
            continue
        seen_outputs.add(sig)
        out.append(info)
    return out


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="규칙 하나의 성질을 계산해 출력")
    ap.add_argument("spec", nargs="?", help="규칙 스펙 또는 프리셋 이름")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--probe", type=int, default=200)
    ap.add_argument("--example", default="7,2,49,13,25")
    ap.add_argument("--list", action="store_true", help="프리셋 목록")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.list or not a.spec:
        for k, v in PRESETS.items():
            print(f"{k:20s} {v}")
        return
    info = analyze(a.spec, a.n, a.m, a.probe)
    rule = Rule(a.spec, a.n)
    ex = tuple(int(v) for v in a.example.split(","))
    if a.json:
        print(json.dumps(info, ensure_ascii=False))
        return
    print(f"규칙: {info['spec']}")
    print(f"예: {list(ex)} -> {list(rule(ex))}")
    print(f"대칭: {info['sym_kind']}  불변군 {info['inv_size']} ({info['inv_label']}), "
          f"등변군 {info['eq_size']} ({info['eq_label']}), 넓은 대칭 {info['permuting_size']}")
    print(f"의존 범위: 평균 {info['dep_mean']:.2f}, 최대 {info['dep_max']}  (출력별 {info['dep_per_output']})")
    print(f"유효 깊이: {info['effective_depth']} / 명목 {info['nominal_depth']}  단계 분류 {info['step_kinds']}")
    print(f"사소한 규칙: {info['trivial']}")


if __name__ == "__main__":
    main()
