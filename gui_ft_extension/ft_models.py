"""Fine-tuned (QE-filtered FT) model entries for the desktop GUI.

These checkpoints are produced by the Section-7 pipeline in this repo:
    pretrained Base/Big v1.1 -> CometKiwi-22 top-1M filter -> 6K FT steps.

NOTE on quality: per Section 7 of the paper, FT'd checkpoints score
1.5-2.3 BLEU LOWER on newstest (the standard MT eval) than their
pretrained baselines. They tend to translate in a more formal,
UN/legislative register because that is what the QE-top of v2 looks
like. We expose them in the GUI for two reasons:

  1. Reproducibility — readers should be able to use the exact
     checkpoints we report on.
  2. Domain experiments — a user translating UN-style content may
     actually prefer the FT register.

The Advanced menu disclaimer in the GUI surfaces this caveat. We do
NOT make these the default model.

Currently no FT checkpoints are uploaded to HuggingFace — the entries
are commented out as a placeholder. Once the FT weights are pushed,
fill in the hf_repo strings and uncomment.
"""

from typing import List

# Imported lazily inside this module to avoid a hard dependency on the
# main app's package layout when this file is read for documentation.
try:
    from gui.model_registry import ModelEntry  # type: ignore
except ImportError:                                # pragma: no cover
    ModelEntry = None  # type: ignore


FT_MODELS: List = []

# Uncomment once FT checkpoints are uploaded to HF. The publish step is in
# scripts/publish_ft_checkpoints.py (TODO).
#
# if ModelEntry is not None:
#     FT_MODELS = [
#         ModelEntry(
#             key="enfr_base_ft",
#             label="English -> French (Base + FT, formal/UN register)",
#             hf_repo="euswbnix/transformer-wmt14-enfr-base-ft",
#             src_lang="en", tgt_lang="fr", arch="base",
#             size_mb_estimate=240,
#             is_ft=True,
#         ),
#         ModelEntry(
#             key="enfr_big_ft",
#             label="English -> French (Big + FT, formal/UN register)",
#             hf_repo="euswbnix/transformer-wmt14-enfr-big-ft",
#             src_lang="en", tgt_lang="fr", arch="big",
#             size_mb_estimate=836,
#             is_ft=True,
#         ),
#     ]
