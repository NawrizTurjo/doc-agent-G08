"""Stage 4 — chunk text (hybrid: rule-based for Plays/Sonnets/Poems, embedding-similarity
semantic chunking for the Biography prose). Ported from the validated Kaggle notebook
codes/chunk_embed_index_shakespeare.ipynb -- see codes/ECI/chunking_strategy_plan.md and
codes/ECI/embed_chunk_index_plan.md for the design rationale this file implements. Chunking
logic itself (regexes, packing, similarity-cut math) is unchanged from that validated
notebook; only the I/O (file reads -> in-memory dict lookups) and the final Chunk-contract
assembly are adapted to this repo's fixed pipeline shape."""
from __future__ import annotations
import re
from collections import Counter, defaultdict

import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger
from ..ingest.loader import load_toc
from .embed import embed_texts

log = get_logger(__name__)

# ============================================================ rule-based (Plays/Sonnets/Poems)
ACT_BODY_RE    = re.compile(r'^\s*ACT\.?\s+([IVXLC0-9]+)\s*\.?\s*$', re.I)
ACT_RUNNING_RE = re.compile(r'\[\s*ACT\s+([IVXLC0-9]+)\s*\.')
SCENE_RE       = re.compile(r'^\s*SCENE\s+([IVXLC0-9JjlI]+)\s*[.,]?\s*[—–-]\s*(.+)$')
SCENE_NOISE_RE = re.compile(r'^\s*SCENE\s+[IVXLC0-9JjlI]+\s*\.\s*[\]\}\)3Jj]\s*$')
SPEAKER_RE     = re.compile(
    r'^((?:[123]|First|Second|Third|Sir|Lady|Mrs?|Old|Young)\.?\s+)?'
    r'([A-Z][a-z]{1,12})\.'
    r'(?:\s+([A-Z][a-z]{1,12})\.)?'
    r'(?=\s|$)')
CATCHWORD_RE   = re.compile(r'\[[A-Za-z][a-z]{0,12}[.?!]?\s*$')
TITLE_RE       = re.compile(r"^[A-Z][A-Z0-9 .,'&;:-]{5,}$")
SONNET_RE      = re.compile(r'^\s*([IVXLC]+)\.\s*$')


def norm_roman(s: str) -> str:
    return s.upper().replace('J', 'I').replace('1', 'I').replace('L', 'I').replace('3', 'III')


def speaker_key(m: re.Match) -> str:
    pre = (m.group(1) or '').strip().rstrip('.')
    return (pre + ' ' + m.group(2)).strip()


def clean_lines(raw: str) -> list[str]:
    out = []
    for ln in raw.splitlines():
        if '[Blank Page]' in ln:
            continue
        if SCENE_NOISE_RE.match(ln):
            continue
        ln = CATCHWORD_RE.sub('', ln).rstrip()
        s = ln.strip()
        if not s:
            continue
        if (TITLE_RE.match(s) and 'SCENE' not in s and 'ACT' not in s
                and not SONNET_RE.match(s)):
            continue
        out.append(ln)
    merged = []
    for ln in out:
        if merged and merged[-1].endswith('-') and ln[:1].islower():
            merged[-1] = merged[-1][:-1] + ln.lstrip()
        else:
            merged.append(ln)
    return merged


def build_vocab(pages: list[tuple[int, str]], min_count: int) -> set[str]:
    c: Counter = Counter()
    for _, raw in pages:
        for ln in clean_lines(raw):
            m = SPEAKER_RE.match(ln)
            if m:
                c[speaker_key(m)] += 1
    return {k for k, v in c.items() if v >= min_count}


def pack(speeches, meta: dict, out: list[dict], target: int, min_words: int) -> None:
    buf, bw, spk = [], 0, []
    for s, t in speeches:
        w = len(t.split())
        if bw + w > target and bw >= min_words:
            out.append(dict(meta, text=' '.join(buf), speakers=list(dict.fromkeys(spk)),
                             n_words=bw, chunk_method='rule_based'))
            buf, bw, spk = [], 0, []
        buf.append('%s: %s' % (s, t))
        bw += w
        spk.append(s)
    if buf:
        out.append(dict(meta, text=' '.join(buf), speakers=list(dict.fromkeys(spk)),
                         n_words=bw, chunk_method='rule_based'))


def chunk_play(work: dict, page_text: dict[int, str], out: list[dict],
               target: int, min_words: int, speaker_min_count: int) -> None:
    pages = [(n, page_text[n]) for n in range(work['start_page'], work['end_page'] + 1)
             if n in page_text]
    if not pages:
        return
    vocab = build_vocab(pages, speaker_min_count)
    act, scene = None, None
    speeches, cur = [], None
    pg0 = pages[0][0]
    pg1 = pages[0][0]

    def flush():
        nonlocal speeches, pg0
        if speeches:
            pack(speeches, dict(title=work['title'], category=work['category'],
                                 act=act, scene=scene, sonnet_num=None,
                                 page_start=pg0, page_end=pg1), out, target, min_words)
        speeches = []

    for n, raw in pages:
        pg1 = n
        # act running header lives in the first 3 raw lines, BEFORE cleaning strips it
        head = '\n'.join(raw.splitlines()[:3])
        mrh = ACT_RUNNING_RE.search(head)
        if mrh:
            act = norm_roman(mrh.group(1))
        for ln in clean_lines(raw):
            mb = ACT_BODY_RE.match(ln)
            if mb:
                if cur:
                    speeches.append(cur); cur = None
                flush()
                act = norm_roman(mb.group(1)); pg0 = n
                continue
            ms = SCENE_RE.match(ln)
            if ms:
                if cur:
                    speeches.append(cur); cur = None
                flush()
                scene = norm_roman(ms.group(1)); pg0 = n
                continue
            m = SPEAKER_RE.match(ln)
            if m and speaker_key(m) in vocab:
                if cur:
                    speeches.append(cur)
                cur = (speaker_key(m), ln[m.end():].strip())
            elif cur:
                cur = (cur[0], (cur[1] + ' ' + ln.strip()).strip())
        if cur:
            speeches.append(cur); cur = None
    flush()


def chunk_sonnets(work: dict, page_text: dict[int, str], out: list[dict]) -> None:
    cur_num, buf = None, []
    for n in range(work['start_page'], work['end_page'] + 1):
        raw = page_text.get(n)
        if raw is None:
            continue
        for ln in clean_lines(raw):
            m = SONNET_RE.match(ln)
            if m:
                if buf:
                    out.append(dict(title=work['title'], category='Poem', act=None,
                                     scene=None, sonnet_num=cur_num, page_start=n, page_end=n,
                                     text=' '.join(buf), speakers=[],
                                     n_words=len(' '.join(buf).split()),
                                     chunk_method='rule_based'))
                cur_num, buf = m.group(1), []
            else:
                buf.append(ln.strip())
    if buf:
        out.append(dict(title=work['title'], category='Poem', act=None, scene=None,
                         sonnet_num=cur_num, page_start=work['end_page'],
                         page_end=work['end_page'], text=' '.join(buf), speakers=[],
                         n_words=len(' '.join(buf).split()), chunk_method='rule_based'))


def chunk_poem(work: dict, page_text: dict[int, str], out: list[dict], target: int) -> None:
    buf, bw, pg0 = [], 0, work['start_page']
    for n in range(work['start_page'], work['end_page'] + 1):
        raw = page_text.get(n)
        if raw is None:
            continue
        for para in re.split(r'\n\s*\n', raw):
            lines = clean_lines(para)
            if not lines:
                continue
            t = ' '.join(l.strip() for l in lines)
            w = len(t.split())
            if bw + w > target and bw > 0:
                out.append(dict(title=work['title'], category='Poem', act=None, scene=None,
                                 sonnet_num=None, page_start=pg0, page_end=n,
                                 text=' '.join(buf), speakers=[], n_words=bw,
                                 chunk_method='rule_based'))
                buf, bw, pg0 = [], 0, n
            buf.append(t); bw += w
    if buf:
        out.append(dict(title=work['title'], category='Poem', act=None, scene=None,
                         sonnet_num=None, page_start=pg0, page_end=work['end_page'],
                         text=' '.join(buf), speakers=[], n_words=bw,
                         chunk_method='rule_based'))


# ============================================================ semantic (Biography + Bacon essay)
BIO_HEADER_RE = re.compile(
    r'^\s*[ivxlc]{0,6}\s*(BIOGRAPHICAL INTRODUCTION\.?|SHAKESPEARE AND BACON\.?)\s*[ivxlc]{0,6}\s*$',
    re.I)
_SPACED_PUNCT_RE = re.compile(r'\s+([;:,.!?])')
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"‘“(])')


def normalize_spacing(text: str) -> str:
    """'word ;' -> 'word;' -- the old-typesetting spaced-punctuation convention seen in the
    bio corpus, which would otherwise confuse a sentence splitter looking for punctuation
    immediately after a word."""
    return _SPACED_PUNCT_RE.sub(r'\1', text)


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def strip_bio_header(raw: str) -> list[str]:
    """Drop the first non-blank line if it's a running header (title +/- roman-numeral page
    marker). Deliberately narrower than the play chunker's TITLE_RE -- bio prose contains
    legitimate all-caps emphasis mid-paragraph (e.g. 'WILLIAM SHAKESPEARE') that must NOT be
    stripped."""
    lines = raw.splitlines()
    body, considered_first = [], False
    for ln in lines:
        if not considered_first and ln.strip():
            considered_first = True
            if BIO_HEADER_RE.match(ln.strip()):
                continue
        body.append(ln)
    return body


def load_bio_sentences(start: int, end: int, bio_page_text: dict[int, str]) -> list[tuple[str, int]]:
    """Concatenate bio_{start:03d}..bio_{end:03d} into one tagged sentence stream --
    processed as ONE document, since sentence similarity crosses page boundaries just like
    speeches do in plays."""
    tagged: list[tuple[str, int]] = []
    for n in range(start, end + 1):
        raw = bio_page_text.get(n)
        if raw is None:
            continue
        body_lines = strip_bio_header(raw)
        text = ' '.join(l.strip() for l in body_lines if l.strip())
        text = re.sub(r'\s+', ' ', text).strip()      # collapse whitespace FIRST
        text = normalize_spacing(text)                 # THEN fix spaced punctuation
        for s in split_sentences(text):
            tagged.append((s, n))
    return tagged


def similarity_cuts(embeddings: np.ndarray, percentile: float) -> list[int]:
    """Indices i such that a cut boundary falls AFTER sentence i. Assumes L2-normalized
    embeddings (dot product == cosine similarity). Threshold is the (100-percentile)th
    percentile of adjacent dissimilarity, computed for THIS document's embeddings only --
    never a fixed global cutoff (chunking_strategy_plan.md S8: 'relative, not a fixed
    threshold')."""
    n = len(embeddings)
    if n < 2:
        return []
    sims = np.array([float(np.dot(embeddings[i], embeddings[i + 1])) for i in range(n - 1)])
    dissim = 1.0 - sims
    threshold = np.percentile(dissim, 100 - percentile)
    return [i for i, d in enumerate(dissim) if d >= threshold]


def segments_from_cuts(n_items: int, cuts: list[int]) -> list[list[int]]:
    cuts = sorted(set(cuts))
    segs, start = [], 0
    for c in cuts:
        segs.append(list(range(start, c + 1)))
        start = c + 1
    if start < n_items:
        segs.append(list(range(start, n_items)))
    return segs


def pack_by_words(n_items: int, word_counts: list[int], target: int) -> list[int]:
    """Fallback safety net: plain sequential word-count packing, used only when a document's
    similarity-cut signal is too flat to produce a sensible number of segments (e.g. a short
    essay may have too little variance for a percentile cutoff to fire meaningfully)."""
    cuts, bw = [], 0
    for i in range(n_items):
        bw += word_counts[i]
        if bw >= target:
            cuts.append(i)
            bw = 0
    return cuts


def enforce_min_words(segments: list[list[int]], word_of, min_words: int) -> list[list[int]]:
    """Same trailing-buffer behavior as pack() above: a leftover buffer under min_words still
    becomes its own final chunk rather than being merged backward -- simpler and more
    predictable than silently growing the previous chunk, and it's what produces the small
    documented rate of under-min chunks (~1.8%, chunking_strategy_plan.md S9)."""
    merged, pending = [], []
    for seg in segments:
        pending = pending + seg
        if sum(word_of(i) for i in pending) >= min_words:
            merged.append(pending); pending = []
    if pending:
        merged.append(pending)
    return merged


def chunk_bio_work(work: dict, bio_page_text: dict[int, str], out: list[dict],
                    target: int, min_words: int, percentile: float, embed_cfg: dict) -> None:
    m_start = re.search(r'bio_(\d+)', work['start_image'])
    m_end = re.search(r'bio_(\d+)', work['end_image'])
    start, end = int(m_start.group(1)), int(m_end.group(1))

    tagged = load_bio_sentences(start, end, bio_page_text)
    if not tagged:
        log.warning(f"[chunk] no sentences found for {work['title']!r} (bio_{start:03d}-{end:03d})")
        return
    sentences = [s for s, _ in tagged]
    pages = [p for _, p in tagged]
    word_counts = [len(s.split()) for s in sentences]

    embeddings = embed_texts(sentences, embed_cfg)
    cuts = similarity_cuts(embeddings, percentile)
    segments = segments_from_cuts(len(sentences), cuts)

    fallback_used = False
    expected_min_segments = max(1, sum(word_counts) // (target * 2))
    if len(segments) < expected_min_segments:
        fallback_used = True
        cuts = pack_by_words(len(sentences), word_counts, target)
        segments = segments_from_cuts(len(sentences), cuts)

    segments = enforce_min_words(segments, lambda i: word_counts[i], min_words)

    for seg in segments:
        text = ' '.join(sentences[i] for i in seg)
        out.append(dict(
            title=work['title'], category='Biography', act=None, scene=None, sonnet_num=None,
            page_start=pages[seg[0]], page_end=pages[seg[-1]],
            text=text, speakers=[], n_words=len(text.split()), chunk_method='semantic',
        ))
    log.info(f"[chunk] {work['title']!r}: {len(sentences)} sentences -> {len(segments)} chunks "
             f"(fallback={'yes' if fallback_used else 'no'})")


# ============================================================ Chunk-contract assembly
def make_page_ids(category: str, page_start: int, page_end: int) -> list[str]:
    if category == 'Biography':
        return [f"bio_{n:03d}" for n in range(page_start, page_end + 1)]
    return [f"page_{n:04d}" for n in range(page_start, page_end + 1)]


def citation_header(c: dict) -> str:
    if c['category'] == 'Biography':
        return f"[{c['title']} | pp.{c['page_start']}-{c['page_end']}]"
    parts = [c['title']]
    if c.get('act'):
        loc = f"Act {c['act']}"
        if c.get('scene'):
            loc += f" Sc {c['scene']}"
        parts.append(loc)
    elif c.get('sonnet_num'):
        parts.append(f"Sonnet {c['sonnet_num']}")
    parts.append(f"pp.{c['page_start']}-{c['page_end']}")
    if c.get('speakers'):
        parts.append(','.join(c['speakers'][:5]))
    return '[' + ' | '.join(parts) + ']'


def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Re-chunk to cfg['index'] size/overlap.

    `chunks` here is the PAGE-granularity output of vision/ocr.transcribe (one Chunk per
    page). This stage re-groups those by work (doc_id) and re-chunks using the hybrid
    method validated in codes/chunk_embed_index_shakespeare.ipynb: rule-based structural
    chunking (ACT/SCENE headers, speaker vocabulary, sonnet numerals) for Plays/Sonnets/
    Poems, packed to ~cfg['index']['chunk_target_words'] words/chunk; embedding-similarity
    cuts for the 2 continuous-prose Biography works. Fixed-size token windows (the starter
    template's original cfg['index']['chunk_tokens']/['overlap'] placeholder) were rejected
    -- see codes/ECI/chunking_strategy_plan.md -- because they'd split mid-speech/mid-scene
    and throw away the free structural metadata this corpus already prints.
    """
    idx_cfg = cfg["index"]
    target = idx_cfg.get("chunk_target_words", 200)
    min_words = idx_cfg.get("chunk_min_words", 40)
    speaker_min_count = idx_cfg.get("speaker_min_count", 2)
    percentile = idx_cfg.get("bio_similarity_percentile", 10)
    toc_path = cfg.get("corpus", {}).get("toc_path", "data/toc.json")

    toc = load_toc(toc_path)
    works_by_id = {str(w["id"]): w for w in toc}

    text_by_doc: dict[str, dict[int, str]] = defaultdict(dict)
    for c in chunks:
        page_id = c.page_ids[0] if c.page_ids else c.id
        n = int(page_id.split('_')[1])
        text_by_doc[c.doc_id][n] = c.text

    chunks_raw: list[dict] = []
    for doc_id, page_text in text_by_doc.items():
        work = works_by_id.get(doc_id)
        if work is None or work["category"] == "Reference":
            continue

        work_chunks: list[dict] = []
        if work["category"] == "Biography":
            chunk_bio_work(work, page_text, work_chunks, target, min_words, percentile, cfg["embed"])
        elif work["category"] in ("Comedy", "History", "Tragedy"):
            chunk_play(work, page_text, work_chunks, target, min_words, speaker_min_count)
        elif work["title"] == "Sonnets":
            chunk_sonnets(work, page_text, work_chunks)
        else:
            chunk_poem(work, page_text, work_chunks, target)

        for wc in work_chunks:
            wc["doc_id"] = doc_id
        chunks_raw.extend(work_chunks)

    log.info(f"[chunk] {len(chunks_raw)} chunks from {len(text_by_doc)} works")

    # ---- hard sanity assertions (ported from the notebook's Part 5 checks) ----
    assert all('[Blank Page]' not in c['text'] for c in chunks_raw), \
        "a chunk contains raw [Blank Page] text"
    assert all(c['category'] != 'Reference' for c in chunks_raw), \
        "a Reference-section chunk leaked in"

    seq_counter: dict[str, int] = defaultdict(int)
    out: list[Chunk] = []
    for c in chunks_raw:
        doc_id = c["doc_id"]
        seq_counter[doc_id] += 1
        chunk_id = f"{doc_id}_{seq_counter[doc_id]:04d}"
        page_ids = make_page_ids(c['category'], c['page_start'], c['page_end'])
        out.append(Chunk(
            id=chunk_id,
            doc_id=doc_id,
            text=citation_header(c) + "\n" + c['text'],
            page_ids=page_ids,
        ))

    assert len({cc.id for cc in out}) == len(out), "duplicate chunk ids!"
    log.info(f"[chunk] built {len(out)} Chunk objects, all ids unique")
    return out
