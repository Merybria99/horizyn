#!/usr/bin/env python3
"""Build the offline HTML deck from the Marp-compatible Markdown source."""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "presentation.md"
DEFAULT_OUTPUT = ROOT / "index.html"


def inline_markup(text: str, source_dir: Path) -> str:
    escaped = html.escape(text, quote=False)

    def image_replace(match: re.Match[str]) -> str:
        alt, raw_path = match.group(1), html.unescape(match.group(2))
        path = (source_dir / raw_path).resolve()
        if path.is_file():
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            src = f"data:{mime};base64,{encoded}"
        else:
            src = html.escape(raw_path, quote=True)
        return f'<img src="{src}" alt="{html.escape(alt, quote=True)}">'

    escaped = re.sub(r"!\[([^]]*)\]\(([^)]+)\)", image_replace, escaped)
    escaped = re.sub(
        r"\[([^]]+)\]\(([^)]+)\)",
        lambda match: (
            f'<a href="{html.escape(html.unescape(match.group(2)), quote=True)}">'
            f"{match.group(1)}</a>"
        ),
        escaped,
    )
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped


def parse_table(lines: list[str], source_dir: Path) -> str:
    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    rows = [cells(line) for line in lines]
    align = rows[1]
    headers = rows[0]
    body = rows[2:]
    classes = ["num" if spec.rstrip().endswith(":") else "" for spec in align]
    head_html = "".join(
        f'<th class="{classes[idx]}">{inline_markup(value, source_dir)}</th>'
        for idx, value in enumerate(headers)
    )
    body_html = "".join(
        "<tr>"
        + "".join(
            f'<td class="{classes[idx] if idx < len(classes) else ""}">'
            f"{inline_markup(value, source_dir)}</td>"
            for idx, value in enumerate(row)
        )
        + "</tr>"
        for row in body
    )
    return f'<div class="table-wrap"><table><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table></div>'


def render_slide(markdown: str, index: int, count: int, source_dir: Path) -> str:
    lines = markdown.strip().splitlines()
    slide_class = ""
    filtered: list[str] = []
    for line in lines:
        match = re.fullmatch(r"<!--\s*_class:\s*([\w-]+)\s*-->", line.strip())
        if match:
            slide_class = match.group(1)
        else:
            filtered.append(line)
    lines = filtered

    output: list[str] = []
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor]
        stripped = line.strip()
        if not stripped:
            cursor += 1
            continue

        if stripped.startswith("```"):
            language = stripped[3:].strip()
            cursor += 1
            code_lines: list[str] = []
            while cursor < len(lines) and not lines[cursor].strip().startswith("```"):
                code_lines.append(lines[cursor])
                cursor += 1
            cursor += 1
            output.append(
                f'<pre data-language="{html.escape(language)}"><code>'
                f"{html.escape(chr(10).join(code_lines))}</code></pre>"
            )
            continue

        if stripped.startswith("|") and cursor + 1 < len(lines):
            separator = lines[cursor + 1].strip()
            if separator.startswith("|") and re.fullmatch(r"[|:\-\s]+", separator):
                table_lines = [line, lines[cursor + 1]]
                cursor += 2
                while cursor < len(lines) and lines[cursor].strip().startswith("|"):
                    table_lines.append(lines[cursor])
                    cursor += 1
                output.append(parse_table(table_lines, source_dir))
                continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            output.append(f"<h{level}>{inline_markup(heading.group(2), source_dir)}</h{level}>")
            cursor += 1
            continue

        list_match = re.match(r"^[-*]\s+(.+)$", stripped)
        ordered_match = re.match(r"^\d+\.\s+(.+)$", stripped)
        if list_match or ordered_match:
            tag = "ol" if ordered_match else "ul"
            items: list[str] = []
            current = (ordered_match or list_match).group(1)
            cursor += 1
            while cursor < len(lines):
                candidate = lines[cursor].strip()
                same = re.match(r"^\d+\.\s+(.+)$", candidate) if tag == "ol" else re.match(r"^[-*]\s+(.+)$", candidate)
                if same:
                    items.append(current)
                    current = same.group(1)
                    cursor += 1
                    continue
                if not candidate:
                    break
                if candidate.startswith(("#", "|", "```")):
                    break
                current += " " + candidate
                cursor += 1
            items.append(current)
            output.append(
                f"<{tag}>"
                + "".join(f"<li>{inline_markup(item, source_dir)}</li>" for item in items)
                + f"</{tag}>"
            )
            continue

        paragraph = [stripped]
        cursor += 1
        while cursor < len(lines):
            candidate = lines[cursor].strip()
            if not candidate or candidate.startswith(("#", "|", "```", "- ", "* ")) or re.match(r"^\d+\.\s+", candidate):
                break
            paragraph.append(candidate)
            cursor += 1
        output.append(f"<p>{inline_markup(' '.join(paragraph), source_dir)}</p>")

    first_heading = next((item for item in output if item.startswith("<h1>")), "")
    title_text = re.sub(r"<[^>]+>", "", first_heading)
    semantic_class = ""
    if "Appendix" in title_text:
        semantic_class = " appendix"
    if any(token in title_text for token in ("Table 1", "F-series", "Q-series", "B-series", "R-series")):
        semantic_class += " evidence"
    return (
        f'<section class="slide {html.escape(slide_class)}{semantic_class}" '
        f'data-slide="{index}" aria-label="Slide {index + 1} of {count}">'
        '<div class="slide-accent"></div><main>'
        + "".join(output)
        + "</main>"
        + f'<footer><span>Horizyn enzyme-reaction retrieval</span><span>{index + 1:02d} / {count:02d}</span></footer>'
        + "</section>"
    )


def split_slides(source: str) -> list[str]:
    lines = source.splitlines()
    if lines and lines[0].strip() == "---":
        closing = next(idx for idx in range(1, len(lines)) if lines[idx].strip() == "---")
        lines = lines[closing + 1 :]
    body = "\n".join(lines)
    return [chunk.strip() for chunk in re.split(r"(?m)^---\s*$", body) if chunk.strip()]


STYLE = r"""
:root {
  --ink: #161a1d;
  --muted: #5c646a;
  --paper: #f7f8f6;
  --white: #ffffff;
  --line: #d7dcda;
  --teal: #087f78;
  --teal-soft: #dcefed;
  --amber: #c66b18;
  --amber-soft: #f8eadc;
  --green: #3f7d51;
  --red: #a33d42;
  --blue: #276c8d;
  --slide-width: 1600px;
  --slide-height: 900px;
}
* { box-sizing: border-box; }
html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: #252a2b; }
body { font-family: Inter, Aptos, "Segoe UI", Arial, sans-serif; color: var(--ink); letter-spacing: 0; }
.deck { position: fixed; inset: 0; }
.slide {
  position: absolute;
  left: 50%; top: 50%;
  width: var(--slide-width); height: var(--slide-height);
  padding: 62px 78px 48px;
  background: var(--paper);
  transform: translate(-50%, -50%) scale(var(--deck-scale, 1));
  transform-origin: center;
  display: none;
  overflow: hidden;
  box-shadow: 0 14px 50px rgba(0, 0, 0, 0.28);
}
.slide.active { display: block; }
.slide::before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image: linear-gradient(rgba(22,26,29,.028) 1px, transparent 1px), linear-gradient(90deg, rgba(22,26,29,.028) 1px, transparent 1px);
  background-size: 40px 40px;
}
.slide-accent { position: absolute; left: 0; top: 0; bottom: 0; width: 10px; background: var(--teal); }
.slide main { position: relative; z-index: 1; height: 750px; overflow: hidden; }
h1 { margin: 0 0 26px; max-width: 1380px; font-size: 52px; line-height: 1.08; font-weight: 760; letter-spacing: 0; }
h2 { margin: 4px 0 24px; color: var(--teal); font-size: 30px; line-height: 1.25; font-weight: 650; letter-spacing: 0; }
h3 { margin: 18px 0 10px; font-size: 24px; color: var(--amber); letter-spacing: 0; }
p, li { font-size: 25px; line-height: 1.42; }
p { margin: 16px 0; max-width: 1380px; }
ul, ol { margin: 12px 0 0 28px; padding-left: 24px; max-width: 1400px; }
li { margin: 8px 0; padding-left: 7px; }
li::marker { color: var(--teal); font-weight: 700; }
strong { color: #064f4b; }
code { font-family: "SFMono-Regular", Consolas, monospace; background: #e9ecea; border: 1px solid #d5d9d7; border-radius: 4px; padding: 1px 6px; font-size: .86em; }
pre { margin: 20px 0; width: 100%; max-width: 1300px; background: #1c2223; color: #eef2ef; border-left: 7px solid var(--amber); padding: 22px 26px; border-radius: 4px; overflow: hidden; }
pre code { border: 0; background: transparent; color: inherit; padding: 0; font-size: 21px; line-height: 1.38; }
.table-wrap { width: 100%; max-height: 645px; overflow: hidden; border: 1px solid var(--line); border-radius: 6px; background: var(--white); }
table { width: 100%; border-collapse: collapse; table-layout: auto; }
th { background: #202727; color: white; font-size: 17px; line-height: 1.25; text-align: left; padding: 12px 13px; vertical-align: bottom; }
td { border-top: 1px solid var(--line); font-size: 17px; line-height: 1.28; padding: 10px 13px; vertical-align: top; }
tbody tr:nth-child(even) { background: #f0f3f1; }
th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.slide.evidence td, .slide.appendix td { font-size: 16px; padding: 8px 11px; }
.slide.evidence th, .slide.appendix th { font-size: 16px; padding: 10px 11px; }
img { display: block; max-width: 900px; max-height: 310px; object-fit: contain; margin: 18px auto 0; border: 1px solid var(--line); background: white; }
a { color: var(--blue); text-decoration-thickness: 2px; }
footer { position: absolute; left: 78px; right: 62px; bottom: 22px; display: flex; justify-content: space-between; color: var(--muted); font-size: 15px; z-index: 2; }
.title { background: #17201f; color: white; padding-top: 210px; }
.title::before { background-image: linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px); }
.title .slide-accent { width: 22px; background: var(--amber); }
.title h1 { font-size: 78px; max-width: 1240px; margin-bottom: 28px; }
.title h2 { color: #8ed2cd; font-size: 34px; max-width: 1100px; }
.title p { color: #c6cecb; font-size: 22px; margin-top: 70px; }
.title footer { color: #a8b3af; }
.section { background: #e6f1ef; padding-top: 285px; }
.section h1 { color: #064f4b; font-size: 74px; }
.section p { font-size: 28px; color: var(--muted); }
.appendix .slide-accent { background: var(--amber); }
.controls { position: fixed; right: 18px; bottom: 16px; display: flex; gap: 8px; z-index: 20; }
.control { width: 42px; height: 42px; border: 1px solid rgba(255,255,255,.3); background: rgba(20,24,25,.86); color: white; border-radius: 5px; cursor: pointer; font-size: 22px; line-height: 1; }
.control:hover { background: var(--teal); }
.progress { position: fixed; left: 0; bottom: 0; height: 5px; width: var(--progress, 0%); background: var(--amber); z-index: 30; transition: width 160ms ease; }
@media (max-width: 700px) { .controls { right: 8px; bottom: 8px; } .control { width: 36px; height: 36px; } }
@media print {
  @page { size: 13.333in 7.5in; margin: 0; }
  html, body { overflow: visible; background: white; }
  .deck { position: static; }
  .slide { position: relative; display: block !important; left: 0; top: 0; transform: none !important; width: 13.333in; height: 7.5in; page-break-after: always; box-shadow: none; }
  .controls, .progress { display: none; }
}
"""


SCRIPT = r"""
(() => {
  const slides = [...document.querySelectorAll('.slide')];
  const deck = document.querySelector('.deck');
  const progress = document.querySelector('.progress');
  let current = Math.max(0, Math.min(slides.length - 1, Number(location.hash.slice(1)) - 1 || 0));
  function fit() {
    const scale = Math.min(innerWidth / 1600, innerHeight / 900);
    deck.style.setProperty('--deck-scale', String(scale));
  }
  function show(index, replace = false) {
    current = Math.max(0, Math.min(slides.length - 1, index));
    slides.forEach((slide, idx) => slide.classList.toggle('active', idx === current));
    progress.style.setProperty('--progress', `${((current + 1) / slides.length) * 100}%`);
    const nextHash = `#${current + 1}`;
    if (location.hash !== nextHash) history[replace ? 'replaceState' : 'pushState'](null, '', nextHash);
    document.title = `${current + 1}/${slides.length} | Enzyme-Reaction Retrieval in Horizyn`;
  }
  document.querySelector('[data-action="prev"]').addEventListener('click', () => show(current - 1));
  document.querySelector('[data-action="next"]').addEventListener('click', () => show(current + 1));
  document.querySelector('[data-action="full"]').addEventListener('click', () => {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen();
    else document.exitFullscreen();
  });
  addEventListener('keydown', event => {
    if (['ArrowRight', 'PageDown', ' '].includes(event.key)) { event.preventDefault(); show(current + 1); }
    if (['ArrowLeft', 'PageUp'].includes(event.key)) { event.preventDefault(); show(current - 1); }
    if (event.key === 'Home') show(0);
    if (event.key === 'End') show(slides.length - 1);
  });
  addEventListener('hashchange', () => show(Number(location.hash.slice(1)) - 1, true));
  addEventListener('resize', fit);
  fit();
  show(current, true);
})();
"""


def build(source_path: Path, output_path: Path) -> int:
    source = source_path.read_text(encoding="utf-8")
    slide_sources = split_slides(source)
    slides = "\n".join(
        render_slide(markdown, index, len(slide_sources), source_path.parent)
        for index, markdown in enumerate(slide_sources)
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Horizyn enzyme-reaction retrieval methods and comparisons">
<title>Enzyme-Reaction Retrieval in Horizyn</title>
<style>{STYLE}</style>
</head>
<body>
<div class="deck">{slides}</div>
<nav class="controls" aria-label="Slide controls">
  <button class="control" data-action="prev" title="Previous slide" aria-label="Previous slide">&#8592;</button>
  <button class="control" data-action="next" title="Next slide" aria-label="Next slide">&#8594;</button>
  <button class="control" data-action="full" title="Toggle fullscreen" aria-label="Toggle fullscreen">&#9974;</button>
</nav>
<div class="progress" aria-hidden="true"></div>
<script>{SCRIPT}</script>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")
    return len(slide_sources)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    count = build(args.source.resolve(), args.output.resolve())
    print(f"Built {count} slides at {args.output.resolve()}")


if __name__ == "__main__":
    main()
