"""Evaluate two abstention rules for the same local memory retriever."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import negatives
from fetch import CACHE

ROOT = Path(__file__).resolve().parent
BUDGET = 10  # False interruptions per 100 no-memory turns, on calibration only.
DRAWS = 10_000
GATES = ("cosine", "margin")


class Encoder:
    def __init__(self):
        from onnxruntime import InferenceSession, SessionOptions
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(CACHE / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        options = SessionOptions()
        options.intra_op_num_threads = 4
        self.session = InferenceSession(
            str(CACHE / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts):
        vectors = []
        for start in range(0, len(texts), 32):
            batch = self.tokenizer.encode_batch(texts[start : start + 32])
            inputs = {
                "input_ids": np.asarray([item.ids for item in batch], dtype=np.int64),
                "attention_mask": np.asarray(
                    [item.attention_mask for item in batch], dtype=np.int64
                ),
                "token_type_ids": np.asarray([item.type_ids for item in batch], dtype=np.int64),
            }
            hidden = self.session.run(["last_hidden_state"], inputs)[0]
            mask = inputs["attention_mask"][..., None].astype(hidden.dtype)
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
            vectors.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True))
        return np.concatenate(vectors)


@dataclass
class Scores:
    positive: np.ndarray
    correct: np.ndarray
    negative: np.ndarray


def split_conversations(rows):
    ordered = sorted(rows, key=lambda row: row["sample_id"])
    return ordered[::2], ordered[1::2]


def conversation_data(row):
    memories = {
        turn["dia_id"]: turn["text"]
        for name, session in row["conversation"].items()
        if name.startswith("session_") and isinstance(session, list)
        for turn in session
        if turn.get("text", "").strip()
    }
    positives = []
    for qa in row["qa"]:
        evidence = set(qa.get("evidence", []))
        if qa["category"] in (1, 2, 3, 4) and len(evidence) == 1 and evidence <= memories.keys():
            positives.append((qa["question"], next(iter(evidence))))
    return memories, positives


def rank(similarities):
    """Break cosine ties by dialogue order. Return winner and both gate scores."""
    indices = np.argsort(-similarities, axis=1, kind="stable")[:, :2]
    top = np.take_along_axis(similarities, indices, axis=1)
    return indices[:, 0], np.column_stack((top[:, 0], top[:, 0] - top[:, 1]))


def score_split(rows, prompts, encoder):
    negative_vectors = encoder.encode(prompts)
    scores = []
    for row in rows:
        memories, positives = conversation_data(row)
        questions, evidence = zip(*positives)
        memory_vectors = encoder.encode(list(memories.values()))
        positive_vectors = encoder.encode(questions)
        winner, positive = rank(positive_vectors @ memory_vectors.T)
        _, negative = rank(negative_vectors @ memory_vectors.T)
        correct = np.asarray(list(memories))[winner] == evidence
        scores.append(Scores(positive, correct, negative))
    return scores


def choose_threshold(positive, correct, negative):
    """Maximize correct surfaces under the budget, breaking ties conservatively."""
    grid = np.arange(1001) / 1000
    hits = (positive[correct, None] > grid).sum(axis=0)
    interruptions = (negative[:, None] > grid).sum(axis=0)
    feasible = np.flatnonzero(100 * interruptions <= BUDGET * len(negative))
    best = max(feasible, key=lambda index: (hits[index], -interruptions[index], index))
    return float(grid[best])


def fit(scores):
    positive = np.concatenate([row.positive for row in scores])
    correct = np.concatenate([row.correct for row in scores])
    negative = np.concatenate([row.negative for row in scores])
    return np.asarray([
        choose_threshold(positive[:, column], correct, negative[:, column])
        for column in range(len(GATES))
    ])


def outcomes(scores, thresholds):
    hits = np.asarray([
        ((row.positive > thresholds) & row.correct[:, None]).sum(axis=0) for row in scores
    ])
    totals = np.asarray([len(row.correct) for row in scores])
    interruptions = np.asarray([row.negative > thresholds for row in scores])
    return hits, totals, interruptions


def summarize(scores, thresholds):
    hits, totals, interruptions = outcomes(scores, thresholds)
    return np.stack((100 * hits.sum(axis=0) / totals.sum(), 100 * interruptions.mean(axis=(0, 1))))


def bootstrap(scores, thresholds):
    """Keep policies paired. Resample conversations and shared negative prompts."""
    hits, totals, interruptions = outcomes(scores, thresholds)
    conversations, prompts, _ = interruptions.shape
    rng = np.random.default_rng(42)
    rows = rng.integers(conversations, size=(DRAWS, conversations))
    columns = rng.integers(prompts, size=(DRAWS, prompts))
    recalls = 100 * hits[rows].sum(axis=1) / totals[rows].sum(axis=1, keepdims=True)
    false_rates = 100 * interruptions[rows[:, :, None], columns[:, None, :]].mean(axis=(1, 2))
    draws = np.stack((recalls, false_rates), axis=1)
    return np.percentile(draws, [2.5, 97.5], axis=0), np.percentile(
        draws[..., 1] - draws[..., 0], [2.5, 97.5], axis=0
    )


def plot(point, intervals, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.7), layout="constrained")
    fig.suptitle("Memory gates on held-out conversations", fontweight="bold")
    colors = ("#147d80", "#99482a")
    panels = ("Evidence recall@1 (%)", "False interruptions per 100 no-memory turns")
    for panel, (ax, title) in enumerate(zip(axes, panels)):
        for gate, color in enumerate(colors):
            y = 1 - gate
            lo, hi = intervals[:, panel, gate]
            ax.hlines(y, lo, hi, color=color, linewidth=2)
            ax.vlines([lo, hi], y - .06, y + .06, color=color, linewidth=1.5)
            ax.scatter(point[panel, gate], y, color=color, s=38, zorder=3)
        ax.set(yticks=[0, 1], yticklabels=["Margin", "Cosine"], ylim=(-.45, 1.45), xlabel=title)
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=.25)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[1].axvline(BUDGET, color="#6a6a6a", ls="--", lw=1)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def print_split(label, rows, scores):
    print(f"{label}: {', '.join(row['sample_id'] for row in rows)}")
    print(f"  {sum(len(row.correct) for row in scores)} positive queries; "
          f"{sum(len(row.negative) for row in scores)} negative evaluations "
          f"({len(scores[0].negative)} distinct prompts)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path, default=ROOT / "results.png")
    parser.add_argument("--ask", help="Try one query against a local conversation instead")
    parser.add_argument("--gate", choices=GATES, default="cosine")
    parser.add_argument("--conversation", default="conv-50", help="Conversation for --ask")
    args = parser.parse_args()
    if not all((CACHE / name).is_file() for name in ("locomo10.json", "model.onnx", "tokenizer.json")):
        parser.error("run python fetch.py first")

    rows = json.loads((CACHE / "locomo10.json").read_bytes())
    calibration, test = split_conversations(rows)
    encoder = Encoder()
    calibration_scores = score_split(calibration, negatives.CALIBRATION, encoder)
    thresholds = fit(calibration_scores)
    if args.ask is not None:
        chosen = next((row for row in rows if row["sample_id"] == args.conversation), None)
        if chosen is None:
            parser.error(f"unknown conversation: {args.conversation}")
        memories, _ = conversation_data(chosen)
        similarities = encoder.encode([args.ask]) @ encoder.encode(list(memories.values())).T
        winner, scores = rank(similarities)
        gate = GATES.index(args.gate)
        score, threshold = scores[0, gate], thresholds[gate]
        print(f"{args.gate}: {score:.4f} (threshold {threshold:.3f})")
        if score > threshold:
            memory_id = list(memories)[winner[0]]
            print(f"{memory_id}: {memories[memory_id]}")
        else:
            print("Memory stayed quiet.")
        return

    test_scores = score_split(test, negatives.TEST, encoder)
    print_split("Calibration", calibration, calibration_scores)
    print_split("Test", test, test_scores)
    calibrated = summarize(calibration_scores, thresholds)
    point = summarize(test_scores, thresholds)
    intervals, differences = bootstrap(test_scores, thresholds)
    for index, name in enumerate(GATES):
        print(f"{name}: threshold {thresholds[index]:.3f}; "
              f"calibration recall {calibrated[0, index]:.2f}%, "
              f"false interruptions/100 {calibrated[1, index]:.2f}")
        for panel, metric in enumerate(("recall@1 (%)", "false interruptions/100")):
            lo, hi = intervals[:, panel, index]
            print(f"  test {metric}: {point[panel, index]:.2f} [{lo:.2f}, {hi:.2f}]")
    for panel, metric in enumerate(("recall@1 (pp)", "false interruptions/100")):
        lo, hi = differences[:, panel]
        print(f"  margin minus cosine {metric}: {point[panel, 1] - point[panel, 0]:+.2f} "
              f"[{lo:+.2f}, {hi:+.2f}]")
    hits, totals, _ = outcomes(test_scores, thresholds)
    ceiling = sum(row.correct.sum() for row in test_scores)
    print(f"Ungated test recall@1: {100 * ceiling / totals.sum():.2f}% ({ceiling}/{totals.sum()})")
    print(f"Test hits: {hits.sum(axis=0).tolist()}")
    plot(point, intervals, args.plot)


if __name__ == "__main__":
    main()
