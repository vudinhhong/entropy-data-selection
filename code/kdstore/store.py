"""Reading an entropy-gated top-K logit store.

The file holds, for every answer position of every sequence in the teacher's
corpus half, the teacher's K largest next-token probabilities, the identities
those probabilities belong to, and the entropy of the full distribution at that
position. The entropy column is what the gate reads; the gate is applied when
the store is *used*, not when it is written, so one file supports any threshold.

Arrays, in file order:

    offsets    [n_sequences + 1]  row span of each sequence
    idx        [n_positions, K]   token identities, int32
    logp       [n_positions, K]   log-probabilities, float16
    H          [n_positions]      entropy of the full distribution, nats
    gold_p     [n_positions]      probability the teacher gave the gold token
    gold_rank  [n_positions]      rank of the gold token

Older stores carry `prob` instead of `logp`. The two cost the same sixteen bits
but are not interchangeable: float16 bottoms out at 6e-8, so storing raw
probabilities erases the tail wherever the teacher is confident, which is most
of the corpus. Log space keeps a ratio of 1e-13 to within a percent.
"""
import math

import numpy as np


class LogitStore:
    """A published store, held in memory.

    `np.load` hands back a lazy object that re-reads and re-inflates the whole
    array on every slice, so a loop over a few hundred sequences turns into
    tens of GB of I/O. Reading once up front costs a few seconds and a few
    hundred MB, and makes everything afterwards fast.
    """

    def __init__(self, path):
        self.path = str(path)
        with np.load(self.path) as z:
            self._arrays = {k: z[k] for k in z.files}
        self._probabilities = None

        missing = {"idx", "H", "offsets"} - set(self._arrays)
        if missing:
            raise ValueError(f"{self.path}: store is missing {sorted(missing)}")
        if "logp" not in self._arrays and "prob" not in self._arrays:
            raise ValueError(f"{self.path}: store has neither 'logp' nor 'prob'")

    # ------------------------------------------------------------- shape

    @property
    def n_positions(self):
        return int(self._arrays["idx"].shape[0])

    @property
    def n_sequences(self):
        return int(self._arrays["offsets"].shape[0] - 1)

    @property
    def k_max(self):
        """The widest view this file supports. Every experiment reads K <= this."""
        return int(self._arrays["idx"].shape[1])

    @property
    def encoding(self):
        return "logp" if "logp" in self._arrays else "prob"

    def __repr__(self):
        return (f"<LogitStore {self.n_sequences} sequences, "
                f"{self.n_positions} positions, K={self.k_max}, "
                f"{self.encoding}>")

    # ------------------------------------------------------------- access

    @property
    def probabilities(self):
        """Top-K probabilities, whichever way the file stores them.

        Converted once and kept, because the gate and the loss both want it.
        """
        if self._probabilities is None:
            if self.encoding == "logp":
                self._probabilities = np.exp(
                    self._arrays["logp"].astype(np.float32))
            else:
                self._probabilities = self._arrays["prob"].astype(np.float32)
        return self._probabilities

    def span(self, sequence_id):
        """Row range (start, stop) that belongs to one sequence."""
        off = self._arrays["offsets"]
        return int(off[sequence_id]), int(off[sequence_id + 1])

    def rows_for(self, sequence_ids):
        """Every row belonging to a list of sequences, in sequence order."""
        off = self._arrays["offsets"]
        spans = [np.arange(off[i], off[i + 1]) for i in sequence_ids]
        if not spans:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(spans)

    def entropy(self, rows=None):
        """Teacher entropy in nats.

        Cast to float64 before anything reduces over it. `np.quantile` on a
        float16 array of a million elements returns nan without warning,
        because it computes the virtual index (n-1)*q in the array's own dtype
        and n-1 overflows float16.
        """
        H = self._arrays["H"]
        H = H if rows is None else H[rows]
        return H.astype(np.float64)

    def gate(self, tau, rows=None, direction="high"):
        """Boolean mask of the positions the gate keeps.

        high  H > tau, where the teacher is uncertain and the distribution has
              something to say beyond the gold token
        low   H < tau, the complement, used as a control
        all   no gate, used to measure the ceiling
        """
        if direction == "all":
            n = self.n_positions if rows is None else len(rows)
            return np.ones(n, dtype=bool)
        H = self.entropy(rows)
        return H < tau if direction == "low" else H > tau

    def view(self, rows, k):
        """The (ids, probabilities) a consumer sees at width k."""
        if k > self.k_max:
            raise ValueError(f"store holds K={self.k_max}, asked for K={k}")
        return (self._arrays["idx"][rows, :k].astype(np.int64),
                self.probabilities[rows, :k])

    def array(self, name):
        """Raw access, for the byte accounting and the integrity checks."""
        return self._arrays[name]

    # ------------------------------------------------------------- summary

    def describe(self, tau=None, k=None):
        """Numbers worth printing before anyone trains on this file."""
        H = self.entropy()
        k = k or self.k_max
        mass = self.probabilities[:, :k].sum(-1)
        out = {
            "path": self.path,
            "encoding": self.encoding,
            "sequences": self.n_sequences,
            "positions": self.n_positions,
            "k_max": self.k_max,
            "entropy_min": float(H.min()),
            "entropy_median": float(np.median(H)),
            "entropy_mean": float(H.mean()),
            "entropy_max": float(H.max()),
            f"top{k}_mass_mean": float(mass.mean()),
        }
        if tau is not None:
            kept = float((H > tau).mean())
            out["tau"] = tau
            out["rho"] = kept
            out["mask_bits_per_position"] = binary_entropy(kept)
        return out


def binary_entropy(rho):
    """h(rho) in bits, the cost of naming which positions passed the gate.

    This is the entropy bound on a membership mask, and it is what B(tau, K)
    charges. It is not a real encoder, but at the few kilobytes involved the
    difference does not move a digit.
    """
    if rho <= 0.0 or rho >= 1.0:
        return 0.0
    return -(rho * math.log2(rho) + (1 - rho) * math.log2(1 - rho))


def storage_cost(n_positions, rho, k, vocab_size, value_bits=16):
    """B(tau, K) in bytes: the mask, plus one identity and one value per entry.

        B = N * h(rho) + rho * N * K * (ceil(log2 V) + b)

    The first term is flat in K; only the second shrinks as the store narrows.
    """
    id_bits = math.ceil(math.log2(vocab_size))
    mask = n_positions * binary_entropy(rho)
    payload = rho * n_positions * k * (id_bits + value_bits)
    return (mask + payload) / 8.0
