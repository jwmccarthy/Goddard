"""TrueSkill and Bradley-Terry ratings from a league: uv run --with trueskill --with matplotlib python tools/league_ratings.py league.json"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import trueskill
from scipy.optimize import minimize
from scipy.stats import spearmanr

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ORDERS = 20
BOOTSTRAP = 200
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def outcomes(games: list[dict]) -> np.ndarray:
    """(blue, orange, score) with score 1 / 0.5 / 0 for blue."""
    return np.array([
        (g["blue"], g["orange"], np.sign(g["blue_goals"] - g["orange_goals"]) / 2 + 0.5)
        for g in games
    ])


def trueskill_ratings(results: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    draw = float(np.mean(results[:, 2] == 0.5))
    env = trueskill.TrueSkill(draw_probability=max(draw, 1e-3))
    rng = np.random.default_rng(0)
    mus, sigmas = [], []
    for _ in range(ORDERS):
        ratings = [env.create_rating() for _ in range(count)]
        for a, b, score in results[rng.permutation(len(results))]:
            a, b = int(a), int(b)
            if score == 0.5:
                ratings[a], ratings[b] = env.rate_1vs1(ratings[a], ratings[b], drawn=True)
            elif score == 1:
                ratings[a], ratings[b] = env.rate_1vs1(ratings[a], ratings[b])
            else:
                ratings[b], ratings[a] = env.rate_1vs1(ratings[b], ratings[a])
        mus.append([r.mu for r in ratings])
        sigmas.append([r.sigma for r in ratings])
    return np.mean(mus, 0), np.mean(sigmas, 0)


def bradley_terry(results: np.ndarray, count: int) -> np.ndarray:
    """Elo-scale strengths (first checkpoint = 0); draws count as half a win."""
    a, b, score = results[:, 0].astype(int), results[:, 1].astype(int), results[:, 2]

    def loss(free):
        theta = np.concatenate(([0.0], free))
        p = 1 / (1 + np.exp(-(theta[a] - theta[b])))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        grad_p = score / p - (1 - score) / (1 - p)
        value = -(score * np.log(p) + (1 - score) * np.log(1 - p)).sum()
        g = np.zeros(count)
        np.add.at(g, a, -grad_p * p * (1 - p))
        np.add.at(g, b, grad_p * p * (1 - p))
        return value + 1e-4 * (free ** 2).sum(), g[1:] + 2e-4 * free

    fit = minimize(loss, np.zeros(count - 1), jac=True, method="L-BFGS-B")
    return np.concatenate(([0.0], fit.x)) * 400 / np.log(10)


def plot(ticks, mu, sigma, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(ticks, mu - sigma, mu + sigma, color=SERIES, alpha=0.18, linewidth=0)
    ax.plot(ticks, mu, color=SERIES, linewidth=2)
    ax.plot(ticks, mu, "o", color=SERIES, markersize=4, markeredgecolor=SURFACE, markeredgewidth=1)
    best = int(np.argmax(mu))
    ax.annotate(f"peak {mu[best]:.1f} at {ticks[best]:.1f}B", (ticks[best], mu[best]),
                xytext=(0, 10), textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel("Training ticks (billions)", color=MUTED)
    ax.set_ylabel("TrueSkill μ (band: ±σ)", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("league", type=Path)
    args = parser.parse_args()
    league = json.loads(args.league.read_text())
    count = len(league["steps"])
    ticks = np.array(league["steps"]) * league["frameskip"] / 1e9
    results = outcomes(league["games"])

    mu, sigma = trueskill_ratings(results, count)
    elo = bradley_terry(results, count)
    rng = np.random.default_rng(1)
    samples = np.array([
        bradley_terry(results[rng.integers(len(results), size=len(results))], count)
        for _ in range(BOOTSTRAP)
    ])
    low, high = np.percentile(samples, [2.5, 97.5], 0)

    goals = np.array([[g["blue_goals"], g["orange_goals"]] for g in league["games"]])
    summary = {
        "games":                               len(results),
        "draw_rate":                           float(np.mean(results[:, 2] == 0.5)),
        "goals_per_game":                      float(goals.sum(1).mean()),
        "blue_score":                          float(results[:, 2].mean()),
        "spearman_trueskill_vs_bradley_terry": float(spearmanr(mu, elo)[0]),
        "checkpoints":                         [
            {"step": s, "ticks_b": float(t), "trueskill_mu": float(m), "trueskill_sigma": float(sd),
             "elo": float(e), "elo_low": float(lo), "elo_high": float(hi)}
            for s, t, m, sd, e, lo, hi in zip(league["steps"], ticks, mu, sigma, elo, low, high)
        ]
    }
    stem = args.league.with_suffix("")
    Path(f"{stem}-ratings.json").write_text(json.dumps(summary, indent=1))
    run = Path(league["run"]).name
    plot(ticks, mu, sigma, Path(f"{stem}-trueskill.png"),
         f"{run}: TrueSkill by checkpoint ({len(results):,} games, {league['seconds']:.0f} s each)")
    print(json.dumps({k: v for k, v in summary.items() if k != "checkpoints"}, indent=1))
    for row in summary["checkpoints"][:: max(1, count // 13)] + summary["checkpoints"][-1:]:
        print(f"{row['ticks_b']:6.1f}B  mu {row['trueskill_mu']:6.2f} ± {row['trueskill_sigma']:.2f}"
              f"   elo {row['elo']:7.0f} [{row['elo_low']:.0f}, {row['elo_high']:.0f}]")


if __name__ == "__main__":
    main()
