# Corpus provenance

- **Source (URL):** *The Complete Works of William Shakespeare*, P.F. Collier & Son edition (New York,
  early 1900s). Original scan: Internet Archive, from a University of California Libraries copy —
  https://archive.org/details/completeworksofw00shakrich . Re-derived page-image dataset (300 DPI PNGs,
  rendered from that source PDF) published on Kaggle for reuse:
  https://www.kaggle.com/datasets/abhishekroy48/complete-works-of-shakespeare
- **Licence / usage rights:** Public domain. Internet Archive's own rights review lists this scan's
  copyright status as `NOT_IN_COPYRIGHT`; no copyright notice appears on any inspected page. Shakespeare
  (d. 1616) and this early-1900s trade edition are both unambiguously public domain. Free to re-share —
  we already re-share the extracted page images ourselves via the Kaggle dataset above.
- **Pages:** 1,374 total in the source PDF (121 MB) → 37 biographical front-matter pages
  (`bio_001`–`bio_037`) + 1,322 main-corpus pages (`page_0001`–`page_1322`, the plays/poems), plus a
  small number of additional front-matter/cover images outside that numbered sequence.
  **Words:** ~1,000,000+ (estimated at ~800 words/page over ~1,300 pages of literary text, from sampled
  OCR) — comfortably clears the course floor (>=300 pages, >=60,000 words).
  **Size on disk:** 4.51 GB (extracted PNG page-image set).
- **Scan/script difficulty notes:** Single-column dense prose (Biographical Introduction) vs. mostly
  two-column justified play/verse text with running headers, page numbers, italicized speaker prefixes
  (e.g. "Fal.", "Host."), and bracketed stage directions (e.g. "[Exit Simple.]"). Old-style typography
  (æ ligatures, archaic punctuation, occasional Latin block quotes) in the biographical section.
  Smaller-font embedded quotes appear within the single-column prose. Each play opens with its own
  cast-list/dramatis-personae layout, distinct from both running prose and two-column dialogue. Scans
  themselves are otherwise clean — no fading or skew observed on any page inspected so far.
- **Split policy (by document):** 70% train / 15% validation / 15% test, split by whole document — every
  page of a given play/essay goes entirely into one split (Comedies, Histories, Tragedies, and the two
  Biography pieces), confirmed against `toc.json`. The individual poems each follow the same
  whole-document rule. Exception: the 154 Sonnets are each self-contained, so the split unit there is one
  sonnet, assigned individually (~70/15/15 by count) rather than the whole section going into one split.
  This also prevents leakage into OCR fine-tuning (A2), since every page of a test-split document stays
  out of fine-tuning data too.
