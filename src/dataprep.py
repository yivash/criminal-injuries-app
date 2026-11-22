"""Prepare the 2012 Criminal Injuries Compensation Scheme (13 June 2019 edition).

Existing app.py imports and artifact names
are retained. Rebuild ALL artifacts after replacing this file, then restart
Streamlit so its cached index/records are discarded.

    python src/dataprep.py --prepare-only
    python src/dataprep.py

--prepare-only writes to indexes/criminal_injuries_preview/, never over live
artifacts. --pdf and --output-dir allow a separate validation/build location.

No LLM is used in preparation.
LegalBertEmbeddings used for 

This is an edition-specific parser, not a general PDF-table parser. The PDF hash,
layout checks, exact row counts, all level/amount pairs, and regression checks
must pass. Source text is retained (in "data" folder) alongside derived hierarchy for auditing.
The current app can search the new records, but complete comparisons and parent
expansion still require the retrieval changes discussed separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from pypdf import PdfReader
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "data" / "criminal-injuries-compensation-scheme-2012.pdf"
INDEX_DIR = ROOT / "indexes" / "criminal_injuries"
CHUNKS_PATH = ROOT / "indexes" / "criminal_injuries_doc_chunks.pkl"
RECORDS_PATH = ROOT / "indexes" / "criminal_injuries_records.json"
TARIFF_PATH = ROOT / "indexes" / "criminal_injuries_tariff.json"
MANIFEST_PATH = ROOT / "indexes" / "criminal_injuries_manifest.json"
PREPARATION_PATH = ROOT / "indexes" / "criminal_injuries_preparation.json"
SECTIONS_PATH = ROOT / "indexes" / "criminal_injuries_sections.json"
DOCUMENT_ID = "criminal-injuries-compensation-scheme-2012-amended-2019"
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
EXPECTED_SHA256 = "a3d57aa551a5d353a5f9c8242fa4bbb831c9eda6b93a4a4370f1457078127b35"
# pipeline_version is the existing app's artifact CONTRACT, not parser revision.
PIPELINE_VERSION = 1
EXTRACTION_VERSION = 2
RULE_PAGES = range(10, 44)
TARIFF_PAGES = {"A": range(56, 77), "B": range(77, 81)}
PRICE = re.compile(r"\b([AB](?:20|1\d|[1-9]))\s+(\d{1,3}(?:,\d{3})+|\d{4,6})\b")
RULE_START = re.compile(r"^(\d{1,3}[A-Z]?)\.\s+")
LEVEL_TABLE = re.compile(r"Level\s+([AB](?:20|1\d|[1-9]))\s+£\s*([\d,]+)")
MULTIPLIER = re.compile(r"(?m)^\s*(\d{1,2})\s+(\d+\.\d{4})\s+(0\.\d{4})\s*$")

# Reviewed top-level headings: these all share the left margin with subheadings,
# so indentation alone cannot distinguish them. Every listed heading must occur.
A_GROUPS = {
    "Burns", "Burns (causing more than minor disfigurement)", "Scarring",
    "Major paralysis (other than as a result of brain damage)",
    "Medically recognised illness or condition (excluding minor and mental injury)",
    "Mental injury", "Peripheral sensory nerve damage",
    "Peripheral motor nerve damage not otherwise compensated for", "Brain Damage",
    "Epilepsy - continuing disability", "Ear", "Eye", "Face", "Neck", "Nose",
    "Skull", "Teeth", "Tongue", "Arm", "Elbow", "Finger and Thumb", "Hand",
    "Humerus (upper arm bone)", "Radius (a forearm bone)", "Shoulder",
    "Damage to one or more of tendon, ligament or cartilage",
    "Ulna (a forearm bone)", "Wrist", "Abdomen", "Back", "Chest",
    "Clavicle (collar bone)", "Coccyx (tail bone)", "Genitalia", "Hernia",
    "Kidney", "Lung", "Pancreas", "Pelvis", "Scapula (shoulder blade)",
    "Spleen", "Sternum (breast bone)", "Ankle", "Femur (thigh bone)",
    "Fibula (slender bone from knee to ankle)", "Foot", "Heel", "Hip", "Knee",
    "Leg", "Tibia (shin bone)", "Toe",
}
REGIONS = {"GENERAL", "HEAD & NECK", "UPPER LIMBS", "TORSO", "LOWER LIMBS"}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")


@dataclass
class Block:
    text: str
    x: float
    y: float
    last_y: float
    page: int
    end_page: int
    italic: bool = False


def page_lines(page, number: int) -> list[Block]:
    """Read content order; join same-baseline fragments (including split words).

    Do not sort by y: many price cells are emitted AFTER their wrapped
    descriptions, despite sharing the first line's y coordinate.
    """
    spans = []

    def visitor(text, cm, tm, font, size):
        if text.strip() and tm[5] > 55:  # body; excludes running page number
            spans.append(Block(text.rstrip("\n"), tm[4], tm[5], tm[5],
                               number, number, "Italic" in str((font or {}).get("/BaseFont"))))

    page.extract_text(visitor_text=visitor)
    lines = []
    for item in spans:
        if (lines and abs(lines[-1].y - item.y) < 1
                and lines[-1].italic == item.italic):
            # Literal spaces in the text runs normally encode word boundaries.
            lines[-1].text += item.text
        else:
            lines.append(replace(item))
    return [replace(line, text=clean(line.text)) for line in lines]


def load_source(path: Path):
    require(path.is_file(), f"Scheme PDF not found: {path}")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_SHA256,
            "Unexpected PDF edition. Do not bypass this check: review the parser for the new PDF.")
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    require(len(pages) == 83 and "as amended on 13 June 2019" in pages[3],
            "Unexpected Scheme page count or amendment date")
    for n in range(10, 83):
        labels = re.findall(r"(?m)^\s*(\d{1,2})\s*$", pages[n - 1])
        require(str(n - 9) in labels, f"PDF page {n}: expected printed-page label missing")
    return reader, pages


def make_record(kind, text, first, last=None, **metadata):
    text = text.strip()
    require(bool(text), f"Empty {kind} record")
    meta = dict(source="data/criminal-injuries-compensation-scheme-2012.pdf",
                document_id=DOCUMENT_ID, scheme_version="2012 as amended 13 June 2019",
                kind=kind, pdf_page=first, pdf_page_end=last or first,
                printed_page=first - 9, printed_page_end=(last or first) - 9, **metadata)
    meta["record_id"] = digest([meta, text])[:24]
    return {"text": text, "metadata": meta}


def referenced_paragraphs(text):
    refs = []
    pattern = r"\bparagraphs?\s+(\d+[A-Z]?(?:(?:\s*,\s*|\s+and\s+|\s+to\s+)\d+[A-Z]?)*)"
    for match in re.finditer(pattern, text, re.I):
        expression = match.group(1)
        for a, b in re.findall(r"\b(\d+)\s+to\s+(\d+)\b", expression):
            refs.extend(str(n) for n in range(int(a), int(b) + 1))
        refs.extend(re.findall(r"\d+[A-Z]?", expression))
    return list(dict.fromkeys(refs))


def rule_records(reader):
    """Headings are detached BEFORE assigning numbered provisions to sections."""
    records, current, section = [], None, "Introduction"
    pending_heading = []

    def finish():
        nonlocal current
        if current:
            body = "\n".join(current["parts"])
            records.append(make_record(
                "rule", f"{current['section']}\n{body}", current["first"], current["last"],
                paragraph=current["number"], section_title=current["section"],
                section_id=slug(current["section"]), raw_text=body,
                referenced_paragraphs=referenced_paragraphs(body)))
            current = None

    for n in RULE_PAGES:
        for line in page_lines(reader.pages[n - 1], n):
            t = line.text
            if t.startswith("The Criminal Injuries Compensation Scheme 2012"):
                continue
            match = RULE_START.match(t)
            # Main headings in this edition sit at x=124.2; continuation heading
            # lines at x=124.7. Subclauses start '(' and must remain with the rule.
            is_heading = (line.x < 130 and not match and re.match(r"[A-Za-z]", t))
            if is_heading:
                finish()
                pending_heading.append(t)
                continue
            if match:
                finish()
                if pending_heading:
                    section = clean(" ".join(pending_heading))
                    pending_heading = []
                current = dict(number=match.group(1), section=section,
                               first=n, last=n, parts=[t])
            elif current:
                current["parts"].append(t)
                current["last"] = n
            else:
                require(not t.strip(". "), f"Unassigned rule text on page {n}: {t}")
    finish()
    require(not pending_heading, "Unassigned trailing rule heading")
    numbers = [r["metadata"]["paragraph"] for r in records]
    expected = {str(n) for n in range(1, 147) if n != 19} | {"2A", "18A", "88A"}
    # Paragraph 19 was omitted by the 2019 amendment and is absent.
    require(len(numbers) == len(set(numbers)), "Duplicate main paragraph identifiers")
    require(set(numbers) == expected,
            f"Main paragraph coverage differs: missing={expected-set(numbers)}, extra={set(numbers)-expected}")
    grouped = defaultdict(list)
    for row in records:
        grouped[row["metadata"]["section_id"]].append(row)
    sections = []
    for sid, rows in grouped.items():
        title = rows[0]["metadata"]["section_title"]
        section = make_record(
            "section", title + "\n\n" + "\n\n".join(r["metadata"]["raw_text"] for r in rows),
            rows[0]["metadata"]["pdf_page"], rows[-1]["metadata"]["pdf_page_end"],
            section_id=sid, section_title=title,
            paragraphs=[r["metadata"]["paragraph"] for r in rows],
            child_record_ids=[r["metadata"]["record_id"] for r in rows])
        sections.append(section)
    # Complete compact sections are searchable, alongside individual provisions.
    records.extend(s for s in sections if len(s["text"]) <= 2400)
    # A procedure window is also useful when the containing section is longer.
    # This range is the source's own contents-table entry (111-113: Procedure).
    by_number = {r["metadata"]["paragraph"]: r for r in records if r["metadata"]["kind"] == "rule"}
    steps = [by_number[str(n)] for n in range(111, 114)]
    records.append(make_record(
        "procedure", "Reconsideration and repayment: procedure\n\n" +
        "\n\n".join(r["metadata"]["raw_text"] for r in steps), 37,
        section_id="reconsideration-and-repayment", section_title="Reconsideration and repayment",
        paragraphs=["111", "112", "113"],
        child_record_ids=[r["metadata"]["record_id"] for r in steps],
        referenced_paragraphs=["109", "110"]))
    return records, sections


def tariff_blocks(reader):
    blocks = []
    for n in range(56, 81):
        page_blocks = []
        for line in page_lines(reader.pages[n - 1], n):
            t = line.text
            if (t.startswith("TARIFF OF INJURIES") or t.startswith("Description of injury")
                    or t in {"Amount £", "PAYMENTS"}):
                continue
            price = PRICE.search(t)
            if price and not t[:price.start()].strip():
                require(bool(page_blocks), f"Orphan tariff price on page {n}")
                require(not PRICE.search(page_blocks[-1].text), f"Two prices assigned to one row on {n}")
                page_blocks[-1].text += " " + t
            elif (page_blocks and not PRICE.search(page_blocks[-1].text)
                  and line.italic == page_blocks[-1].italic
                  and abs(line.x - page_blocks[-1].x) < 5
                  and 0 < page_blocks[-1].last_y - line.y < 19.2):
                page_blocks[-1].text += " " + t
                page_blocks[-1].last_y = line.y
            else:
                page_blocks.append(replace(line))
        blocks.extend(page_blocks)
    # Reviewed cross-page prose continuations. A price can occur on the first
    # page while its description continues on the next. Preserve both pages.
    continuations = {
        (57, "persist for 28 weeks or more from the incident or date of onset"),
        (70, "or more organs"), (71, "inhalation"),
    }
    merged, seen = [], set()
    for block in blocks:
        key = (block.page, block.text)
        if key in continuations:
            require(bool(merged), "Missing row before continuation")
            previous = merged[-1]
            price = PRICE.search(previous.text)
            if price:
                previous.text = previous.text[:price.start()].rstrip() + " " + block.text + " " + price.group()
            else:
                previous.text += " " + block.text
            previous.end_page = block.end_page
            seen.add(key)
        else:
            merged.append(block)
    require(seen == continuations, "Expected cross-page tariff continuations changed")
    # PDF p.79 places the mental-illness qualifier and its first priced child at
    # effectively the same indentation; split this explicitly, never inherit
    # the first child's 'moderate' qualifier into the next 'severe' row.
    fixed = []
    special = 0
    for b in merged:
        marker = " - moderate mental illness"
        if b.page == 79 and "psychiatric prognosis" in b.text and marker in b.text:
            parent, child = b.text.split(marker, 1)
            fixed.append(replace(b, text=parent, x=175.0))
            fixed.append(replace(b, text="- moderate mental illness" + child, x=190.0))
            special += 1
        elif b.page == 79 and b.x > 170 and b.text.startswith("- severe me ntal illness"):
            fixed.append(replace(b, x=190.0))
        elif b.page == 80 and b.text.startswith("Where a person has sustained"):
            prefix, pregnancy = b.text.rsplit(" Pregnancy ", 1)
            fixed.append(replace(b, text=prefix, italic=True))
            fixed.append(replace(b, text="Pregnancy " + pregnancy))
        else:
            fixed.append(b)
    require(special == 1, "Expected page-79 qualifier layout changed")
    return fixed


def level_amounts(pages):
    pairs = LEVEL_TABLE.findall(pages[51])
    amounts = {code: int(value.replace(",", "")) for code, value in pairs}
    require(len(pairs) == len(amounts) == 35, "Tariff level table incomplete/duplicated")
    require(set(amounts) == {f"A{i}" for i in range(1, 21)} | {f"B{i}" for i in range(1, 16)},
            "Tariff level codes differ from expected")
    require(amounts["A3"] == 1800 and amounts["B3"] == 2000 and amounts["A20"] == 250000,
            "Tariff control amounts changed")
    return amounts


def tariff_records(reader, pages, amounts):
    """Maintain region/category + unpriced ancestor stack, never prior priced row."""
    entries, notes = [], []
    region, category, subcategory, stack = "", "", "", []
    category_page, region_page, subcategory_page = 56, 56, 56
    seen_groups, note_current = set(), None
    ordinal = Counter()
    for block in tariff_blocks(reader):
        t, x, page = clean(block.text), block.x, block.page
        part = "A" if page < 77 else "B"
        if page == 77 and region != "PART B":
            region, category, subcategory, stack = "PART B", "", "", []
            region_page = page
        if block.italic:
            note_match = re.match(r"Note \[(\d+)\]:", t)
            if note_match:
                note_current = make_record("tariff_note", t, page, block.end_page,
                                           annex="E", note_id=note_match.group(1), tariff_part=part)
                notes.append(note_current)
            elif re.match(r"Note \[\d+\] applies", t):
                note_current = None  # scope links are assigned below
            elif t.startswith("Where a person has sustained"):
                notes.append(make_record("tariff_note", t, page, block.end_page,
                                         annex="E", note_id="other-payments", tariff_part=part))
                note_current = None
            else:
                require(note_current is not None, f"Unassigned italic tariff prose on {page}: {t}")
                note_current["text"] += "\n\n" + t
                note_current["metadata"]["pdf_page_end"] = block.end_page
                note_current["metadata"]["printed_page_end"] = block.end_page - 9
            continue
        note_current = None
        if t in REGIONS:
            region, region_page = t, page
            category, subcategory, stack = "", "", []
            continue
        matches = list(PRICE.finditer(t))
        is_b_group = (t == "Fatal criminal injury" or t.startswith("Physical abuse of ")
                      or t.startswith("Sexual offence where victim") or t == "Other payments")
        is_a_group = t in A_GROUPS
        # Within the head/neck burns and scarring groups, Face and Neck are
        # subheadings, not the later independent Face/Neck injury groups.
        if page == 58 and t in {"Face", "Neck"}:
            is_a_group = False
        if not matches and (is_a_group or is_b_group):
            category, category_page = t, page
            subcategory, stack = "", []
            if is_a_group:
                seen_groups.add(t)
            continue
        if not matches and t == "Patella (knee cap)":
            subcategory, subcategory_page, stack = t, page, []
            continue
        require(bool(category), f"Tariff row without category on {page}: {t}")
        if not matches:
            # Indentation jitter is up to 3 points within the same hierarchy.
            stack = [node for node in stack if node.x < x - 5]
            stack.append(block)
            continue
        require(len(matches) == 1, f"Multiple tariff prices in a single row on {page}: {t}")
        match = matches[0]
        code, raw_amount = match.groups()
        amount = int(raw_amount.replace(",", ""))
        require(code.startswith(part) and amounts.get(code) == amount, f"Invalid {code}/{amount} on {page}")
        description = clean(t[:match.start()])
        require(description and not t[match.end():].strip(), f"Malformed priced row: {t}")
        # A left-margin bullet in the children's minor-abuse row belongs under
        # its left-margin 'Minor abuse' heading despite the visual indentation.
        effective_x = 124 if page == 77 and description.startswith("- isolated or intermittent") else x
        stack = [node for node in stack if node.x < effective_x - 5]
        ancestors = [category] + ([subcategory] if subcategory else []) + [clean(node.text) for node in stack]
        path = ([region] if region and region != "PART B" else []) + ancestors
        note_ids = []
        if category.startswith("Burns"):
            note_ids = ["1"]
        elif category == "Mental injury":
            note_ids = ["2"]
        elif category == "Brain Damage":
            note_ids = ["3"] + (["4"] if any("Very serious brain injury" in a for a in ancestors) else [])
        elif category.startswith("Physical abuse"):
            note_ids = ["5"]
        elif category.startswith("Sexual offence"):
            note_ids = ["6"]
        elif category == "Other payments" and (description == "Pregnancy" or any(
                "Sexually transmitted infection other than" in a for a in ancestors)):
            note_ids = ["other-payments"]
        ordinal[page] += 1
        evidence_pages = sorted(set([page, block.end_page, category_page] +
                                   ([region_page] if region and region != "PART B" else []) +
                                   ([subcategory_page] if subcategory else []) +
                                   [p for node in stack for p in (node.page, node.end_page)]))
        text = (f"Annex E Part {part}. Injury tariff / standard compensation amount.\n"
                + " > ".join(path) + "\nQualifying description: " + description +
                f"\nLevel {code}: £{amount:,}." +
                ("\nApplicable tariff notes: " + ", ".join(note_ids) + "." if note_ids else ""))
        entries.append(make_record(
            "tariff_entry", text, min(evidence_pages), max(evidence_pages), tariff_part=part,
            level=code, amount_gbp=amount, category=category, injury_path=path,
            description=description, preceding_context=" > ".join(path),
            note_ids=note_ids, entry_on_page=ordinal[page], row_pdf_page=page,
            row_pdf_page_end=block.end_page, evidence_pages=evidence_pages, raw_text=t))
        # Crucially: the priced row is NOT pushed onto the ancestor stack.
    require(seen_groups == A_GROUPS, f"Missing tariff category headings: {A_GROUPS-seen_groups}")
    # Refresh note IDs after multi-paragraph notes were assembled.
    notes = [make_record("tariff_note", n["text"], n["metadata"]["pdf_page"],
                         n["metadata"]["pdf_page_end"], annex="E",
                         note_id=n["metadata"]["note_id"], tariff_part=n["metadata"]["tariff_part"])
             for n in notes]
    note_lookup = {n["metadata"]["note_id"]: n for n in notes}
    require(set(note_lookup) == {"1", "2", "3", "4", "5", "6", "other-payments"}, "Missing tariff notes")
    # Independent reconciliation against plain extraction: every priced source
    # occurrence must appear exactly once, on the same page, in the same order.
    expected = [(p, code, int(value.replace(",", ""))) for p in range(56, 81)
                for code, value in PRICE.findall(pages[p - 1])]
    actual = [(r["metadata"]["row_pdf_page"], r["metadata"]["level"], r["metadata"]["amount_gbp"])
              for r in entries]
    require(actual == expected, "Tariff geometry and plain-text row coverage disagree")
    for row in entries:
        require(set(row["metadata"]["note_ids"]).issubset(note_lookup), "Broken tariff note reference")
    return entries, notes


def maximum_records(entries):
    """Precomputed searchable maxima; computation always uses the FULL tariff."""
    records = []
    for part in ("A", "B", "all"):
        selected = [r for r in entries if part == "all" or r["metadata"]["tariff_part"] == part]
        maximum = max(r["metadata"]["amount_gbp"] for r in selected)
        winners = [r for r in selected if r["metadata"]["amount_gbp"] == maximum]
        scope = "all Part A and Part B entries" if part == "all" else f"Part {part}"
        text = (f"Highest / maximum standard injury tariff payout across {scope}: £{maximum:,}.\n"
                "This compares standard tariff entries, not the maximum overall award including other payments.\n\n"
                + "\n\n".join(r["text"] for r in winners))
        records.append(make_record(
            "tariff_summary", text, min(r["metadata"]["pdf_page"] for r in winners),
            max(r["metadata"]["pdf_page_end"] for r in winners),
            operation="maximum", scope=part, amount_gbp=maximum,
            entry_record_ids=[r["metadata"]["record_id"] for r in winners],
            compared_entry_count=len(selected)))
    return records


def build_records(reader, pages):
    records, sections = rule_records(reader)
    amounts = level_amounts(pages)
    entries, notes = tariff_records(reader, pages, amounts)
    records += entries + notes + maximum_records(entries)
    for code, amount in amounts.items():
        records.append(make_record("tariff_level",
                       f"Annex E Part {code[0]}, level {code}: standard amount £{amount:,}.", 52,
                       tariff_part=code[0], level=code, amount_gbp=amount))
    # Annex 44 is a contents page, not Annex A. Preserve complete multi-page
    # annexes so continuation pages retain their subject.
    for annex, title, start, end in [("A", "Interpretation", 45, 45),
                                     ("B", "Crime of Violence", 46, 47),
                                     ("C", "Relevant Places", 48, 48),
                                     ("D", "Previous Convictions", 49, 51)]:
        text = "\n\n".join("\n".join(b.text for b in page_lines(reader.pages[p - 1], p))
                           for p in range(start, end + 1))
        records.append(make_record("annex", f"Annex {annex}: {title}\n{text}", start, end,
                                   annex=annex, section_title=title))
    for p in (81, 82):
        for year, multiplier, discount in MULTIPLIER.findall(pages[p - 1]):
            records.append(make_record("multiplier",
                f"Annex F: {year} years of loss; Table A multiplier {multiplier}; "
                f"Table B discount factor {discount}.", p, annex="F",
                years_of_loss=int(year), multiplier=multiplier, discount_factor=discount))
    validate_records(records, sections)
    return records, amounts, sections


def validate_records(records, sections):
    ids = [r["metadata"]["record_id"] for r in records]
    require(len(ids) == len(set(ids)), "Duplicate record IDs")
    entries = [r for r in records if r["metadata"]["kind"] == "tariff_entry"]
    counts = Counter(r["metadata"]["tariff_part"] for r in entries)
    require(counts == {"A": 341, "B": 52}, f"Unexpected tariff row counts: {dict(counts)}")
    years = [r["metadata"]["years_of_loss"] for r in records if r["metadata"]["kind"] == "multiplier"]
    require(sorted(years) == list(range(1, 61)), "Multiplier years missing/duplicated")
    rules = {r["metadata"]["paragraph"]: r for r in records if r["metadata"]["kind"] == "rule"}
    require("Reconsideration and repayment" not in rules["108"]["text"], "Heading attached to preceding rule")
    require(all(rules[str(n)]["metadata"]["section_id"] == "reconsideration-and-repayment"
                for n in range(109, 114)), "Repayment section incomplete")
    require("30 days" in rules["112"]["text"], "Repayment representations deadline lost")
    winners = [r for r in entries if r["metadata"]["amount_gbp"] == 250000]
    require(len(winners) == 2, "Expected two maximum injury tariffs")
    require(any("Quadraplegia/tetraplegia" in r["text"] and
                "substantially incomplete" not in r["text"] for r in winners), "Paralysis qualifier contaminated")
    require(any("Very serious brain injury" in r["text"] and "4" in r["metadata"]["note_ids"]
                for r in winners), "Brain injury maximum lost heading or exception note")
    require(any("Hemiplegia" in r["text"] and r["metadata"]["description"] == "- severe"
                and r["metadata"]["level"] == "A18" and "burns" not in r["text"].casefold()
                for r in entries), "Hemiplegia heading contaminated")
    for row in entries:
        require("Note [" not in row["metadata"]["description"], "Note merged into priced injury")
    require(all(set(s["metadata"]["child_record_ids"]).issubset(ids) for s in sections), "Broken section children")


def chunk_prefix(meta):
    if meta["kind"] == "rule":
        return f"{meta['section_title']} | Paragraph {meta['paragraph']}\n"
    if meta["kind"] == "tariff_entry":
        return f"Annex E Part {meta['tariff_part']} | " + " > ".join(meta["injury_path"]) + "\n"
    if meta.get("section_title"):
        return meta["section_title"] + "\n"
    if meta.get("note_id"):
        return f"Annex E tariff note {meta['note_id']}\n"
    return ""


def make_chunks(records, tokenizer=None):
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunks = []
    for row in records:
        meta, text = row["metadata"], row["text"]
        prefix = chunk_prefix(meta)
        count = (lambda s: len(tokenizer.encode(s, add_special_tokens=False))) if tokenizer else len
        budget = 480 if tokenizer else 1800
        available = budget - count(prefix) - 8
        require(available >= 80, f"Context prefix too long: {meta['record_id']}")
        if count(text) <= budget:
            parts = [text]
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=available, chunk_overlap=min(50, available // 5),
                length_function=count, separators=["\n\n", "\n", ". ", "; ", " ", ""])
            parts = [prefix + part for part in splitter.split_text(text)]
        for i, part in enumerate(parts):
            if tokenizer:
                require(len(tokenizer.encode(part, add_special_tokens=True)) <= 512,
                        f"Chunk exceeds LegalBERT limit: {meta['record_id']}:{i}")
            # Raw text is available in JSON, not copied into every pickled chunk.
            cm = {k: v for k, v in meta.items() if k != "raw_text"}
            cm.update(chunk_id=f"{meta['record_id']}:{i:03d}", chunk_index=i, chunk_count=len(parts))
            chunks.append(Document(page_content=part, metadata=cm))
    require(len({c.metadata["chunk_id"] for c in chunks}) == len(chunks), "Duplicate chunks")
    return chunks


class LegalBertEmbeddings(Embeddings):
    """Same checkpoint/mean-pooling geometry as the original app, now batched.

    Attention-mask pooling excludes batch padding; special tokens are included,
    as in the original single-input mean. No normalisation/distance change.
    Both query and document encodings use this class. Oversized input fails
    explicitly instead of silently dropping its ending.
    """
    def __init__(self, model_name=MODEL_NAME, batch_size=16, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_documents(self, texts):
        output = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
            if tokens["input_ids"].shape[1] > 512:
                raise ValueError("LegalBERT input exceeds 512 tokens; split the document/query first")
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with self.torch.inference_mode():
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            output.extend(vectors.cpu().tolist())
        return output

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_payload(directory, records, amounts, sections, pages, manifest):
    write_json(directory / RECORDS_PATH.name, records)
    entries = [r for r in records if r["metadata"]["kind"] == "tariff_entry"]
    write_json(directory / TARIFF_PATH.name, dict(
        source_sha256=EXPECTED_SHA256, extraction_version=EXTRACTION_VERSION,
        amounts_gbp=amounts, entries=entries,
        notes=[r for r in records if r["metadata"]["kind"] == "tariff_note"],
        summaries=[r for r in records if r["metadata"]["kind"] == "tariff_summary"]))
    write_json(directory / SECTIONS_PATH.name, sections)
    write_json(directory / "criminal_injuries_source_pages.json", [
        {"pdf_page": i, "text": text} for i, text in enumerate(pages, 1)])
    write_json(directory / PREPARATION_PATH.name, manifest)


def publish(staging, destination):
    """Publish a fully built generation; manifest is removed first/written last.

    Existing cached Streamlit sessions must be stopped before this operation.
    This prevents a new loader seeing a valid manifest during partial publication;
    it is not a multi-file transaction or a hot-reload protocol.
    """
    destination.mkdir(parents=True, exist_ok=True)
    (destination / MANIFEST_PATH.name).unlink(missing_ok=True)
    for path in staging.rglob("*"):
        if path.is_file() and path.name != MANIFEST_PATH.name:
            target = destination / path.relative_to(staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)
    os.replace(staging / MANIFEST_PATH.name, destination / MANIFEST_PATH.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--output-dir", type=Path, default=INDEX_DIR.parent)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="Optional torch device: cpu, cuda, etc.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    reader, pages = load_source(args.pdf)
    records, amounts, sections = build_records(reader, pages)
    manifest = dict(document_id=DOCUMENT_ID, source_sha256=EXPECTED_SHA256,
                    pipeline_version=PIPELINE_VERSION, extraction_version=EXTRACTION_VERSION,
                    embedding_model=MODEL_NAME, embedding_pooling="attention_masked_mean_including_special_tokens",
                    embedding_normalized=False, record_count=len(records), records_sha256=digest(records),
                    kinds=dict(Counter(r["metadata"]["kind"] for r in records)))
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        preview = destination / "criminal_injuries_preview"
        preview.mkdir(parents=True, exist_ok=True)
        write_payload(preview, records, amounts, sections, pages, manifest)
        print(f"Validated {len(records)} records, 393 tariff entries, 35 levels and 60 multipliers.")
        print(f"Preview JSON: {preview}. Live JSON/indexes were not changed.")
        return
    from langchain_community.vectorstores import FAISS

    embeddings = LegalBertEmbeddings(batch_size=args.batch_size, device=args.device)
    chunks = make_chunks(records, embeddings.tokenizer)
    print(f"Validated {len(records)} records; embedding {len(chunks)} chunks...")
    with tempfile.TemporaryDirectory(prefix=".scheme-build-", dir=destination) as temporary:
        staging = Path(temporary)
        write_payload(staging, records, amounts, sections, pages, manifest)
        store = FAISS.from_documents(chunks, embeddings, ids=[c.metadata["chunk_id"] for c in chunks])
        store.save_local(str(staging / INDEX_DIR.name))
        with (staging / CHUNKS_PATH.name).open("wb") as handle:
            pickle.dump(chunks, handle, protocol=pickle.HIGHEST_PROTOCOL)
        manifest.update(chunk_count=len(chunks), chunk_ids_sha256=hashlib.sha256(
            "\n".join(c.metadata["chunk_id"] for c in chunks).encode()).hexdigest(),
            chunk_content_sha256=digest([c.page_content for c in chunks]))
        write_json(staging / MANIFEST_PATH.name, manifest)
        require(store.index.ntotal == len(chunks), "FAISS index count differs from chunks")
        require(set(store.index_to_docstore_id.values()) == {c.metadata["chunk_id"] for c in chunks},
                "FAISS IDs differ from chunk IDs")
        publish(staging, destination)
    print(f"Published FAISS, BM25 chunks and structured records under {destination}.")
    print("Restart Streamlit / clear its cached resources before querying the new index.")


if __name__ == "__main__":
    main()
