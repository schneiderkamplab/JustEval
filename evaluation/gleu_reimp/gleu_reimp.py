# gleu_reimp.py
# Custom Hugging Face Evaluate metric module for GLEU (re-implementation via `pip install gleu`)

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional, Dict, Any

import evaluate
import datasets


_DESCRIPTION = """
Corpus-level GLEU (Napoles et al.) computed using the Python re-implementation by Shota Koyama
(https://github.com/shotakoyama/gleu). This metric is commonly used for Grammatical Error Correction (GEC).

Inputs:
- sources: list of original/source sentences (system input)
- predictions: list of system outputs (hevaluatypotheses)
- references: list of lists of reference corrections (multiple references per source)

Returns:
- gleu: float in [0, 1]
"""

_CITATION = r"""
@misc{koyama_gleu_reimplementation,
  author = {Shota Koyama},
  title  = {gleu: Re-implementation of GLEU, evaluation metric of grammatical error correction},
  year   = {2023},
  howpublished = {\url{https://github.com/shotakoyama/gleu}}
}
"""


@dataclass
class _GleuDeps:
    # Small container for imported dependency modules/functions
    set_tokenization: Any
    make_id_rindex: Any
    make_hdrn_accum: Any
    make_dx_xlen: Any
    drn_accum_to_d_rmax: Any
    drn_accum_to_n_accum: Any
    rindex_to_rhlen: Any
    n_accum_to_gleu: Any


class GleuReimp(evaluate.Metric):
    """Hugging Face Evaluate custom metric: GLEU via `pip install gleu`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._deps: Optional[_GleuDeps] = None

    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=(
                "Computes corpus-level GLEU for GEC given sources, predictions, and multiple references."
            ),
            features=datasets.Features(
                {
                    "sources": datasets.Value("string"),
                    "predictions": datasets.Value("string"),
                    # list of references per example
                    "references": datasets.Sequence(datasets.Value("string")),
                }
            ),
        )

    def _download_and_prepare(self, dl_manager: evaluate.DownloadManager) -> None:
        """
        Called once when the metric is loaded/prepared.
        We use this to verify the external dependency (`gleu`) is available and
        to cache imported symbols for faster compute.
        """
        try:
            # Dependency: https://github.com/shotakoyama/gleu  (pip install gleu)
            from gleu.count import set_tokenization
            from gleu.util import make_id_rindex
            from gleu.aggreg import make_hdrn_accum
            from gleu.count import make_dx_xlen
            from gleu.score import (
                drn_accum_to_d_rmax,
                drn_accum_to_n_accum,
                rindex_to_rhlen,
                n_accum_to_gleu,
            )
        except Exception as e:
            raise ImportError(
                "gleu_reimp requires the external package `gleu`.\n"
                "Install it with: pip install gleu\n"
                "Upstream repo: https://github.com/shotakoyama/gleu\n"
                f"Original import error: {type(e).__name__}: {e}"
            ) from e

        self._deps = _GleuDeps(
            set_tokenization=set_tokenization,
            make_id_rindex=make_id_rindex,
            make_hdrn_accum=make_hdrn_accum,
            make_dx_xlen=make_dx_xlen,
            drn_accum_to_d_rmax=drn_accum_to_d_rmax,
            drn_accum_to_n_accum=drn_accum_to_n_accum,
            rindex_to_rhlen=rindex_to_rhlen,
            n_accum_to_gleu=n_accum_to_gleu,
        )

    def _compute(
        self,
        sources: Sequence[str],
        predictions: Sequence[str],
        references: Sequence[Sequence[str]],
        *,
        max_n: int = 4,
        iterations: int = 500,
        tokenization: str = "word",   # "word" or "char"
        max_reference: bool = False,  # if True, choose best reference per sentence (like `gleu -m`)
        fix_seed: bool = False,       # if True, reproduce original reference sampling behavior
    ) -> Dict[str, float]:
        """
        Computes corpus-level GLEU in [0, 1].

        - If max_reference=True: selects the best reference per sentence (no sampling).
        - Else: samples references for `iterations` and averages (standard GLEU+ style).
        """
        if self._deps is None:
            # In case someone constructs the class directly without load()
            self._download_and_prepare(dl_manager=None)  # type: ignore

        d = self._deps
        assert d is not None

        if len(sources) != len(predictions) or len(sources) != len(references):
            raise ValueError(
                "sources, predictions, and references must have the same length "
                f"(got {len(sources)}, {len(predictions)}, {len(references)})."
            )

        if tokenization not in ("word", "char"):
            raise ValueError("tokenization must be 'word' or 'char'.")

        # Configure the global tokenizer used by `gleu`'s n-gram counter.
        d.set_tokenization(tokenization)

        # Re-shape inputs to match the upstream library’s expected structures:
        # rs_dat: list of tuples (refs for each example)
        rs_dat = [tuple(refs) for refs in references]
        if any(len(r) == 0 for r in rs_dat):
            raise ValueError("Each example must have at least 1 reference.")

        # hs_dat: list of tuples (hyps for each example). We have a single system output per example.
        hs_dat = [(h,) for h in predictions]

        # Length tables (used for brevity penalty aggregation) as upstream does
        dr_rlen = d.make_dx_xlen(rs_dat)   # shape: [D, R]
        dh_hlen = d.make_dx_xlen(hs_dat)   # shape: [D, 1]

        # Build accumulators for H=1 hypothesis across dataset:
        # make_hdrn_accum returns a list of H elements; each is a list over D; each contains R x N accumulators.
        h_dats = [predictions]  # H=1
        hdrn_accum = d.make_hdrn_accum(max_n, list(sources), rs_dat, h_dats)

        # Extract single system output accumulators
        drn_accum = hdrn_accum[0]          # list over D
        d_hlen = dh_hlen[:, 0]             # hyp lengths per example

        if max_reference:
            # Choose best reference per example (sentence-level max, then corpus aggregate)
            d_rmax = d.drn_accum_to_d_rmax(drn_accum, dr_rlen, d_hlen)
            n_accum = d.drn_accum_to_n_accum(drn_accum, d_rmax)
            rlen, hlen = d.rindex_to_rhlen(dr_rlen, d_hlen, d_rmax)
            gleu = float(d.n_accum_to_gleu(n_accum, rlen, hlen))
            return {"gleu": gleu}

        # Sampling mode (standard GLEU behavior with multiple refs)
        id_rindex = d.make_id_rindex(iterations, len(dr_rlen), dr_rlen.shape[1], fix_seed)

        # Average over iterations
        gleus = []
        for d_rindex in id_rindex:
            n_accum = d.drn_accum_to_n_accum(drn_accum, d_rindex)
            rlen, hlen = d.rindex_to_rhlen(dr_rlen, d_hlen, d_rindex)
            gleus.append(float(d.n_accum_to_gleu(n_accum, rlen, hlen)))

        gleu = sum(gleus) / len(gleus) if gleus else 0.0
        return {"gleu": gleu}
