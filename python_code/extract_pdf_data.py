# -*- coding: utf-8 -*-

import os
import re
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import pdfplumber
import pandas as pd

# Optional OCR deps (only used if --force-ocr)
try:
    import pytesseract
    from pytesseract import Output as TesseractOutput
    from pdf2image import convert_from_path
    from PIL import Image
except Exception:
    pytesseract = None
    TesseractOutput = None
    convert_from_path = None
    Image = None


# Honorific
title_match_pattern = re.compile(
    r"(Mr\.\s*|H\.R\.H\.\s*|Mx\.\s*|St\.|Miss\ |Mlle\ |Mine\ |H\.H\.\s*|Ind\.\s*|His\ |Ind\ |Ms\ |Mr\ |Sra\ |Sr\ |M\ |On\ |Fr\ |H\.O\.\s*|Rev\ |Mme\ |Msgr\ |On\.\s*|Fr\.\s*|Rev\.\s*|"
    r"H\.E(?:\.\s*(?:Ms\.\s*|Mr\.\s*|Ms\ |Mr\ |Sra\ |Sr\ |Sra\.\s*|Mme|Sr\.\s*|Msgr\.\s*))?|"
    r"Msgr\.\s*|Mrs\.\s*|Sra\.\s*|Sr\.\s*|Ms\.\s*|Dr\.\s*|Prof\.\s*|M\.\s*|Mme|Ms|"
    r"S\.E(?:\.\s*(?:Ms\.\s*|Mr\.\s*|Mme|Mr|Ms|Dr|Msgr\.\s*|M\.\s*|Ms\ |Mr\ |Sra\ |Sr\ |M\ |Sra\.\s*|Sr\.\s*))?)"
)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def split_honorific_and_person(line: str) -> Tuple[str, str]:
    """Split 'Ms. Nabila GHAZLI' -> ('Ms.', 'Nabila GHAZLI')"""
    s = normalize_ws(line)
    m = title_match_pattern.match(s)
    if m:
        honorific = m.group().strip()
        person = s[len(m.group()):].strip()
        return honorific, person
    return "", s


# Core record type
@dataclass
class Record:
    Delegation: str
    Honorific: str
    Person_Name: str
    Affiliation: str
    Status: str  # Parties / Observers / ""


# CoP1–CoP19 extraction helpers
def is_all_caps_header(text: str) -> bool:
    """
    Heuristic for delegation headers:
    allow uppercase letters, spaces, and slashes.
    e.g., "BAHAMAS", "SWITZERLAND / SUISSE / SUIZA"
    """
    if not text:
        return False
    s = text.strip()
    return bool(s and all(c.isupper() or c in " /" for c in s))


def group_chars_to_lines(chars: List[Dict], y_tol: float = 3.0) -> List[Tuple[str, float, float, bool]]:
    """
    Cluster characters into lines, reconstruct text, and flag bold.
    Returns list of (text, y0, y1, is_bold).
    """
    if not chars:
        return []
    chars = sorted(chars, key=lambda c: (c.get("top", 0.0), c.get("x0", 0.0)))
    lines: List[Tuple[List[Dict], float, float]] = []
    cur = [chars[0]]
    y0 = y1 = chars[0].get("top", 0.0)

    for c in chars[1:]:
        top = c.get("top", 0.0)
        if abs(top - y1) <= y_tol:
            cur.append(c)
            y1 = top
        else:
            lines.append((cur, y0, y1))
            cur, y0, y1 = [c], top, top

    lines.append((cur, y0, y1))

    out: List[Tuple[str, float, float, bool]] = []
    for line_chars, y0, y1 in lines:
        line_chars.sort(key=lambda c: c.get("x0", 0.0))
        text = "".join(c.get("text", "") for c in line_chars).strip()
        is_bold = any("Bold" in (c.get("fontname") or "") for c in line_chars)
        out.append((text, y0, y1, is_bold))
    return out


def group_words_to_lines_with_y(words: List[Dict], y_tol: float = 3.0) -> List[Tuple[str, float, float]]:
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w.get("top", 0.0), w.get("x0", 0.0)))
    lines = []
    cur = [words_sorted[0]]
    y0 = y1 = words_sorted[0].get("top", 0.0)

    for w in words_sorted[1:]:
        top = w.get("top", 0.0)
        if abs(top - y1) <= y_tol:
            cur.append(w)
            y1 = top
        else:
            cur.sort(key=lambda ww: ww.get("x0", 0.0))
            txt = " ".join(ww.get("text", "") for ww in cur)
            lines.append((txt, y0, y1))
            cur = [w]
            y0 = y1 = top

    cur.sort(key=lambda ww: ww.get("x0", 0.0))
    txt = " ".join(ww.get("text", "") for ww in cur)
    lines.append((txt, y0, y1))
    return lines


def collect_paragraphs_with_y(lines_with_y: List[Tuple[str, float, float]], para_factor: float = 1.5) -> List[Tuple[List[str], float, float]]:
    if not lines_with_y:
        return []
    mids = [(y0 + y1) / 2.0 for _, y0, y1 in lines_with_y]
    diffs = [mids[i + 1] - mids[i] for i in range(len(mids) - 1)]
    if not diffs:
        return [([lines_with_y[0][0]], lines_with_y[0][1], lines_with_y[0][2])]

    median_gap = sorted(diffs)[len(diffs) // 2]
    thresh = median_gap * para_factor

    paras: List[Tuple[List[str], float, float]] = []
    cur_lines: List[str] = []
    block_y0: Optional[float] = None
    block_y1: Optional[float] = None
    prev_mid: Optional[float] = None

    for (txt, y0, y1), mid in zip(lines_with_y, mids):
        if prev_mid is not None and (mid - prev_mid) > thresh and cur_lines:
            paras.append((cur_lines, block_y0 if block_y0 is not None else y0, block_y1 if block_y1 is not None else y1))
            cur_lines = []
            block_y0 = block_y1 = None

        if not cur_lines:
            block_y0 = y0
        cur_lines.append(txt)
        block_y1 = y1
        prev_mid = mid

    if cur_lines:
        paras.append((cur_lines, block_y0 if block_y0 is not None else 0.0, block_y1 if block_y1 is not None else 0.0))

    return paras


def extract_singlecol_textpdf(pdf_path: str, status_hint: str = "") -> List[Record]:
    records: List[Record] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            lines = group_chars_to_lines(page.chars, y_tol=3.0)
            i = 0
            current_delegation: Optional[str] = None

            while i < len(lines):
                text, _, _, is_bold = lines[i]

                if is_bold and text:
                    current_delegation = text.split("/", 1)[0].strip()
                    i += 1
                    continue

                if current_delegation and text:
                    honorific, person = split_honorific_and_person(text)
                    i += 1

                    affiliation_lines: List[str] = []
                    while i < len(lines) and (not lines[i][3]) and lines[i][0].strip():
                        affiliation_lines.append(lines[i][0].strip())
                        i += 1

                    affiliation = normalize_ws(" ".join(affiliation_lines))
                    records.append(
                        Record(
                            Delegation=current_delegation,
                            Honorific=honorific,
                            Person_Name=person,
                            Affiliation=affiliation,
                            Status=status_hint,
                        )
                    )
                else:
                    i += 1
    return records


def _twocol_records_from_words(words: List[Dict], x0_thresh: float, status_hint: str = "") -> List[Record]:
    records: List[Record] = []

    full_lines = group_words_to_lines_with_y(words, y_tol=3.0)
    full_paras = collect_paragraphs_with_y(full_lines, para_factor=1.5)

    headers: List[Tuple[float, str]] = []
    for blk, y0, y1 in full_paras:
        if len(blk) == 1 and is_all_caps_header(blk[0]):
            nm = blk[0].split("/", 1)[0].strip()
            mid = (y0 + y1) / 2.0
            headers.append((mid, nm))
    headers.sort(key=lambda x: x[0])

    def header_for_mid(mid: float) -> Optional[str]:
        ans = None
        for m, name in headers:
            if m <= mid:
                ans = name
            else:
                break
        return ans

    left_words = sorted([w for w in words if w.get("x0", 0.0) < x0_thresh], key=lambda w: (w.get("top", 0.0), w.get("x0", 0.0)))
    right_words = sorted([w for w in words if w.get("x0", 0.0) >= x0_thresh], key=lambda w: (w.get("top", 0.0), w.get("x0", 0.0)))

    for col_words in (left_words, right_words):
        lined = group_words_to_lines_with_y(col_words, y_tol=3.0)
        paras = collect_paragraphs_with_y(lined, para_factor=1.5)

        for blk, y0, y1 in paras:
            if len(blk) == 1 and is_all_caps_header(blk[0]):
                continue
            if not blk:
                continue

            name_line = normalize_ws(blk[0])
            honorific, person = split_honorific_and_person(name_line)
            affiliation = normalize_ws(" ".join(ln.strip() for ln in blk[1:]))

            mid = (y0 + y1) / 2.0
            delegation = header_for_mid(mid) or ""

            records.append(
                Record(
                    Delegation=delegation,
                    Honorific=honorific,
                    Person_Name=person,
                    Affiliation=affiliation,
                    Status=status_hint,
                )
            )
    return records


def extract_twocol_textpdf(pdf_path: str, x0_thresh: float = 260.0, status_hint: str = "") -> List[Record]:
    out: List[Record] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words() or []
            if not words:
                continue
            out.extend(_twocol_records_from_words(words, x0_thresh=x0_thresh, status_hint=status_hint))
    return out


def ocr_page_to_words(img: "Image.Image") -> List[Dict]:
    data = pytesseract.image_to_data(img, output_type=TesseractOutput.DICT)
    words: List[Dict] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        try:
            conf = int(data.get("conf", ["-1"] * n)[i])
        except Exception:
            conf = -1
        if not txt or conf < 0:
            continue
        left = float(data.get("left", [0] * n)[i])
        top = float(data.get("top", [0] * n)[i])
        words.append({"text": txt, "x0": left, "top": top})
    return words


def extract_with_ocr(
    pdf_path: str,
    x0_thresh: float = 260.0,
    dpi: int = 300,
    poppler_path: Optional[str] = None,
    tesseract_cmd: Optional[str] = None,
    status_hint: str = "",
) -> List[Record]:
    if pytesseract is None or convert_from_path is None:
        raise RuntimeError("OCR requested but pytesseract/pdf2image/PIL are not available in this environment.")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    out: List[Record] = []
    for img in images:
        words = ocr_page_to_words(img)
        if not words:
            continue
        out.extend(_twocol_records_from_words(words, x0_thresh=x0_thresh, status_hint=status_hint))
    return out


def detect_layout_quick(pdf_path: str, x0_thresh: float = 260.0) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:2]:
                words = page.extract_words() or []
                if not words:
                    continue
                left = sum(1 for w in words if w.get("x0", 0.0) < x0_thresh)
                right = sum(1 for w in words if w.get("x0", 0.0) >= x0_thresh)
                total = left + right
                if total == 0:
                    continue
                if left / total >= 0.25 and right / total >= 0.25:
                    return "two"
            return "one"
    except Exception:
        return "one"


def infer_status_from_filename(pdf_path: str) -> str:
    nm = os.path.basename(pdf_path).lower()
    if "observer" in nm or "observers" in nm:
        return "Observers"
    if "part" in nm or "parties" in nm:
        return "Parties"
    return ""


def extract_cites(
    pdf_path: str,
    layout: str = "auto",
    x0_thresh: float = 260.0,
    force_ocr: bool = False,
    tesseract_cmd: Optional[str] = None,
    poppler_path: Optional[str] = None,
    ocr_dpi: int = 300,
    status_hint: str = "",
) -> List[Record]:
    if not status_hint:
        status_hint = infer_status_from_filename(pdf_path)

    if force_ocr:
        return extract_with_ocr(
            pdf_path=pdf_path,
            x0_thresh=x0_thresh,
            dpi=ocr_dpi,
            poppler_path=poppler_path,
            tesseract_cmd=tesseract_cmd,
            status_hint=status_hint,
        )

    mode = detect_layout_quick(pdf_path, x0_thresh=x0_thresh) if layout == "auto" else layout
    if mode == "two":
        return extract_twocol_textpdf(pdf_path, x0_thresh=x0_thresh, status_hint=status_hint)
    return extract_singlecol_textpdf(pdf_path, status_hint=status_hint)


# CoP20 table extractor
_COP20_PARTIES_RE = re.compile(r"\bPARTIES\b", re.IGNORECASE)
_COP20_OBSERVERS_RE = re.compile(r"\bOBSERVERS\b", re.IGNORECASE)

def _clean_cell(x) -> str:
    if x is None:
        return ""
    return normalize_ws(str(x).replace("\n", " "))


def _looks_like_noise_row(delegation: str, name_raw: str) -> bool:
    d = normalize_ws(delegation)
    n = normalize_ws(name_raw)

    if not d or not n:
        return True

    if d.isdigit() or n.isdigit():
        return True

    junk_starts = (
        "Original language",
        "CONVENTION",
        "OF WILD",
        "________________",
        "Twentieth",
        "Vingti",
        "Vigésima",
        "Samarkand",
        "Samarcande",
        "Samarcanda",
        "LIST OF",
        "LISTE",
        "LISTA",
        "[1611",
        "participant",
    )
    if any(n.startswith(j) for j in junk_starts):
        return True
    if any(d.startswith(j) for j in junk_starts):
        return True

    # Section headings
    if _COP20_PARTIES_RE.search(d) and ("/" in d or "PARTES" in d.upper()):
        return True
    if _COP20_OBSERVERS_RE.search(d) and ("/" in d or "OBSERVADORES" in d.upper()):
        return True

    # Must contain letters
    if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", d) or not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", n):
        return True

    return False


def _make_table_settings_safe() -> Dict:
    """
    Some pdfplumber versions are strict about TableSettings keys.
    Keep this minimal + compatible (NO keep_blank_chars).
    """
    return {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "intersection_tolerance": 5,
        "edge_min_length": 3,
        "text_tolerance": 3,
        "text_x_tolerance": 2,
        "text_y_tolerance": 2,
    }


def extract_cop20_tables(pdf_path: str, debug: bool = False, debug_lines_path: Optional[str] = None) -> pd.DataFrame:
    table_settings = _make_table_settings_safe()

    debug_rows = []
    out_records: List[Record] = []

    current_status = "Parties"  # default until first section header

    with pdfplumber.open(pdf_path) as pdf:
        for p_idx, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables(table_settings=table_settings) or []
            except TypeError:
                tables = page.extract_tables(table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines"}) or []

            for t_idx, table in enumerate(tables, start=1):
                if not table:
                    continue

                for r_idx, row in enumerate(table, start=1):
                    cells = [_clean_cell(c) for c in (row or [])]
                    cells = [c for c in cells if c != ""]

                    if not cells:
                        continue
                    if len(cells) == 1:
                        txt = cells[0]
                        if _COP20_PARTIES_RE.search(txt):
                            current_status = "Parties"
                        elif _COP20_OBSERVERS_RE.search(txt):
                            current_status = "Observers"
                        if debug:
                            debug_rows.append(
                                {"page": p_idx, "table": t_idx, "row": r_idx, "delegation": "", "name_raw": txt, "status": current_status, "kept": 0, "reason": "section_or_singleton"}
                            )
                        continue

                    delegation = cells[0]
                    name_raw = " ".join(cells[1:])  

                    if _COP20_PARTIES_RE.search(delegation) and ("/" in delegation or "PARTES" in delegation.upper()):
                        current_status = "Parties"
                        if debug:
                            debug_rows.append(
                                {"page": p_idx, "table": t_idx, "row": r_idx, "delegation": delegation, "name_raw": name_raw, "status": current_status, "kept": 0, "reason": "section_header_row"}
                            )
                        continue
                    if _COP20_OBSERVERS_RE.search(delegation) and ("/" in delegation or "OBSERVADORES" in delegation.upper()):
                        current_status = "Observers"
                        if debug:
                            debug_rows.append(
                                {"page": p_idx, "table": t_idx, "row": r_idx, "delegation": delegation, "name_raw": name_raw, "status": current_status, "kept": 0, "reason": "section_header_row"}
                            )
                        continue

                    if _looks_like_noise_row(delegation, name_raw):
                        if debug:
                            debug_rows.append(
                                {"page": p_idx, "table": t_idx, "row": r_idx, "delegation": delegation, "name_raw": name_raw, "status": current_status, "kept": 0, "reason": "noise"}
                            )
                        continue

                    honorific, person = split_honorific_and_person(name_raw)

                    if not honorific and len(person.split()) < 2:
                        if debug:
                            debug_rows.append(
                                {"page": p_idx, "table": t_idx, "row": r_idx, "delegation": delegation, "name_raw": name_raw, "status": current_status, "kept": 0, "reason": "not_person_like"}
                            )
                        continue

                    out_records.append(
                        Record(
                            Delegation=delegation,
                            Honorific=honorific,
                            Person_Name=person,
                            Affiliation="",  
                            Status=current_status,
                        )
                    )

                    if debug:
                        debug_rows.append(
                            {"page": p_idx, "table": t_idx, "row": r_idx, "delegation": delegation, "name_raw": name_raw, "status": current_status, "kept": 1, "reason": ""}
                        )

    df = pd.DataFrame([r.__dict__ for r in out_records])
    df = df.drop_duplicates().reset_index(drop=True)

    if debug and debug_lines_path:
        pd.DataFrame(debug_rows).to_csv(debug_lines_path, index=False, encoding="utf-8-sig")

    return df


# Output
def to_dataframe(records: List[Record]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in records], columns=["Delegation", "Honorific", "Person_Name", "Affiliation", "Status"])


def write_output(df: pd.DataFrame, out_path: str) -> None:
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")




def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    pdf_path = args.pdf
    out_path = args.out

    debug_lines_path = None
    if args.debug:
        base, ext = os.path.splitext(out_path)
        debug_lines_path = base + ".DEBUG_lines.csv"

    # Auto-detect CoP20 if not explicitly provided
    auto_is_cop20 = ("cop20" in os.path.basename(pdf_path).lower()) or ("cop20" in pdf_path.lower())
    use_cop20 = bool(args.cop20 or auto_is_cop20)

    if use_cop20:
        df = extract_cop20_tables(pdf_path, debug=args.debug, debug_lines_path=debug_lines_path)
        write_output(df, out_path)
        print(f"Wrote {len(df)} rows to {out_path}")
        if args.debug:
            print(f"Wrote debug lines CSV to: {debug_lines_path}")
        return

    # CoP1–19 path
    records = extract_cites(
        pdf_path=pdf_path,
        layout=args.layout,
        x0_thresh=args.x_threshold,
        force_ocr=args.force_ocr,
        tesseract_cmd=args.tesseract_cmd,
        poppler_path=args.poppler_path,
        ocr_dpi=args.ocr_dpi,
        status_hint=args.status_hint,
    )
    df = to_dataframe(records).drop_duplicates().reset_index(drop=True)
    write_output(df, out_path)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
