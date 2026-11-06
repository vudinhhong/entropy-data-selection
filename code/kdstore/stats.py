"""The statistics the paper reports, written out so they can be checked.

Nothing here needs SciPy. The Student-t tail comes from the regularised
incomplete beta function, evaluated with the usual continued fraction; the
binomial tail for McNemar is summed exactly.
"""
import math


# --------------------------------------------------------------- standard error

def se_bound(n, as_percent=True):
    """The standard error of an accuracy on n items, at its worst case.

    sqrt(a(1-a)/n) is largest at a = 0.5, so sqrt(0.25/n) bounds it whatever
    the model scores. The paper uses the bound instead of the realised value
    on purpose: a threshold that moves with the measurement is a threshold that
    can be argued with after the fact. For GSM8K (n = 1319) it is 1.38 points
    and for MATH (n = 500) it is 2.24.
    """
    se = math.sqrt(0.25 / n)
    return se * 100 if as_percent else se


# --------------------------------------------------------------- Student's t

def _betacf(a, b, x, iterations=200, eps=3e-16):
    """Continued fraction for the incomplete beta function."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """Two-sided p-value for Student's t with df degrees of freedom."""
    t = abs(float(t))
    if df <= 0:
        return float("nan")
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def paired_t_test(differences):
    """One-sample t-test on paired differences.

    The unit is the training subset or fold, not the test item, so the test
    generalises across the data the method was given. With five folds the
    degrees of freedom are four and the two-sided critical value is 2.776.
    """
    d = [float(x) for x in differences]
    n = len(d)
    if n < 2:
        raise ValueError("need at least two differences")
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0.0:
        t = float("inf") if mean != 0.0 else 0.0
        p = 0.0 if mean != 0.0 else 1.0
    else:
        t = mean / (sd / math.sqrt(n))
        p = t_sf(t, n - 1)
    return {"mean": mean, "sd": sd, "n": n, "df": n - 1, "t": t, "p": p,
            "se": sd / math.sqrt(n)}


# --------------------------------------------------------------- McNemar

def mcnemar(b, c):
    """Exact two-sided McNemar on the discordant pairs.

    b and c count the items one arm got right and the other wrong, each way
    round. Useful as a second reading, but it is not the paper's primary test:
    pooling it across subsets reuses the same test items, which makes it
    anti-conservative. The unit of generalisation is the subset.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n": n, "p": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return {"b": b, "c": c, "n": n, "p": min(1.0, 2.0 * tail)}


def discordant(a_flags, b_flags):
    """Count (b, c) from two per-item correctness vectors."""
    b = sum(1 for x, y in zip(a_flags, b_flags) if x and not y)
    c = sum(1 for x, y in zip(a_flags, b_flags) if y and not x)
    return b, c


# --------------------------------------------------------------- correlation

def pearson(xs, ys):
    """Pearson r with a p-value and a Fisher-z 95% interval."""
    n = len(xs)
    if n != len(ys) or n < 3:
        raise ValueError("need at least three paired points")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return {"r": float("nan"), "n": n, "p": float("nan"), "ci": (None, None)}
    r = sxy / math.sqrt(sxx * syy)
    r = max(-0.999999999, min(0.999999999, r))
    t = r * math.sqrt((n - 2) / (1 - r * r))
    p = t_sf(t, n - 2)
    z = 0.5 * math.log((1 + r) / (1 - r))
    half = 1.959963985 / math.sqrt(n - 3) if n > 3 else float("inf")
    lo, hi = math.tanh(z - half), math.tanh(z + half)
    return {"r": r, "n": n, "p": p, "t": t, "ci": (lo, hi)}


# --------------------------------------------------------------- validity

def qualifies(a_base, a_full, n_eval):
    """The two-sided validity criterion from the paper.

    A cell is only readable if fine-tuning on the whole corpus moves the base
    model by more than twice the evaluation standard error. Below that, the
    measurement has no room to show a difference between arms, so a null there
    says nothing about the method.
    """
    threshold = 2.0 * se_bound(n_eval)
    gain = a_full - a_base
    return {"gain": gain, "threshold": threshold, "qualified": gain > threshold}
