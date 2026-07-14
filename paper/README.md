# Paper

ACL-format write-up of the Agora / Tokenomics results. Layout follows the
official [acl-style-files](https://github.com/acl-org/acl-style-files)
template; `acl.sty` and `acl_natbib.bst` are vendored verbatim from that repo.

Build (needs a TeX distribution with pdflatex + bibtex):

```bash
cd paper
latexmk -pdf acl_latex.tex        # or: pdflatex, bibtex, pdflatex, pdflatex
```

- `\usepackage[review]{acl}` — anonymous review version (line numbers).
  Switch to `[final]` for camera-ready or `[preprint]` for arXiv.
- Content sections: §2 Problem/Domain (what Agora is), §3 Methods (the sweep,
  the de-confounded control, the strategy-ceiling anchors, the trust probe,
  the ground-truth deception scorer), §4 Initial Results (numbers match the
  locked 2026-07 analysis on the site), Limitations, and a reproduction
  appendix.
- Numbers in §4 come from `docs/samples/gradient/*.json` and the run
  transcripts under `runs/qwen/`; regenerate with
  `scripts/refresh_site_data.py` before updating the paper.
