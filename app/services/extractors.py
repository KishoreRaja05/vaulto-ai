import csv
import io
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import HTTPException

from app.services.statement_llm import (
    MAX_TRANSACTIONS,
    extract_transactions_with_llm,
    parse_amount,
    parse_date_dayfirst,
    validate_transactions,
)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 15
MAX_ROWS = 500

DATE_HEADERS = {
    "date",
    "txn date",
    "transaction date",
    "value date",
    "posting date",
    "trans date",
}
DESCRIPTION_HEADERS = {
    "description",
    "narration",
    "particulars",
    "details",
    "remarks",
    "merchant",
    "transaction details",
}
AMOUNT_HEADERS = {"amount", "txn amount", "transaction amount", "amt"}
DEBIT_HEADERS = {"debit", "withdrawal", "withdrawals", "dr", "debit amount"}
CREDIT_HEADERS = {"credit", "deposit", "deposits", "cr", "credit amount"}
TYPE_HEADERS = {"type", "transaction type", "dr/cr", "drcr", "cr/dr"}


def normalize_header(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_file_type(data: bytes) -> str:
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if data.startswith(b"%PDF"):
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"PK\x03\x04") or data.startswith(b"PK\x05\x06") or data.startswith(b"PK\x07\x08"):
        if _looks_like_xlsx(data):
            return "xlsx"
        raise HTTPException(status_code=415, detail="Unsupported file type.")
    if _looks_like_csv(data):
        return "csv"
    raise HTTPException(status_code=415, detail="Unsupported file type.")


def _looks_like_xlsx(data: bytes) -> bool:
    try:
        import zipfile

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
        return "[Content_Types].xml" in names and any(
            name.startswith("xl/") or name == "xl/workbook.xml" for name in names
        )
    except Exception:
        return False


def _looks_like_csv(data: bytes) -> bool:
    sample = data[:4096]
    if b"\x00" in sample:
        return False
    try:
        text = sample.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = sample.decode("cp1252")
        except UnicodeDecodeError:
            return False
    if not text.strip():
        return False
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\r\n\t")
    if printable / max(len(text), 1) < 0.9:
        return False
    try:
        dialect = csv.Sniffer().sniff(text, delimiters=[",", ";", "\t", "|"])
        if dialect.delimiter in {",", ";", "\t", "|"} and ("\n" in text or "\r" in text):
            return True
    except Exception:
        pass
    return ("," in text or "\t" in text or ";" in text) and ("\n" in text or "\r" in text)


def map_columns(columns: List[Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for col in columns:
        norm = normalize_header(col)
        if not norm:
            continue
        if norm in DATE_HEADERS and "date" not in mapping:
            mapping["date"] = str(col)
        elif norm in DESCRIPTION_HEADERS and "description" not in mapping:
            mapping["description"] = str(col)
        elif norm in AMOUNT_HEADERS and "amount" not in mapping:
            mapping["amount"] = str(col)
        elif norm in DEBIT_HEADERS and "debit" not in mapping:
            mapping["debit"] = str(col)
        elif norm in CREDIT_HEADERS and "credit" not in mapping:
            mapping["credit"] = str(col)
        elif norm in TYPE_HEADERS and "type" not in mapping:
            mapping["type"] = str(col)
    return mapping


def _derive_type_and_amount(row: pd.Series, mapping: Dict[str, str]) -> Tuple[Optional[str], Optional[float]]:
    if "debit" in mapping or "credit" in mapping:
        debit = parse_amount(row.get(mapping.get("debit"))) if "debit" in mapping else None
        credit = parse_amount(row.get(mapping.get("credit"))) if "credit" in mapping else None
        if debit is not None and abs(debit) > 0:
            return "expense", abs(debit)
        if credit is not None and abs(credit) > 0:
            return "income", abs(credit)
        return None, None

    if "amount" not in mapping:
        return None, None
    amount = parse_amount(row.get(mapping.get("amount")))
    if amount is None:
        return None, None

    tx_type = None
    if "type" in mapping:
        raw_type = str(row.get(mapping.get("type")) or "").strip().lower()
        if raw_type in {"credit", "cr", "deposit", "income"}:
            tx_type = "income"
        elif raw_type in {"debit", "dr", "withdrawal", "expense"}:
            tx_type = "expense"

    if tx_type is None:
        tx_type = "income" if amount < 0 and "type" not in mapping else ("expense" if amount >= 0 else "income")
    # If amount column carries sign, prefer absolute amount and infer if type not explicit.
    if "type" not in mapping and amount < 0:
        tx_type = "income"
    return tx_type, abs(amount)


def rows_from_dataframe(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], bool]:
    if df is None or df.empty:
        return [], False
    # Drop fully empty columns/rows and normalize columns
    df = df.dropna(how="all")
    if df.empty:
        return [], False

    mapping = map_columns(list(df.columns))
    has_date = "date" in mapping
    has_desc = "description" in mapping
    has_amountish = "amount" in mapping or "debit" in mapping or "credit" in mapping
    if not (has_date and has_amountish):
        return [], False

    rows: List[Dict[str, Any]] = []
    for _, row in df.head(MAX_ROWS).iterrows():
        date_val = parse_date_dayfirst(row.get(mapping["date"]))
        tx_type, amount = _derive_type_and_amount(row, mapping)
        if not date_val or amount is None or amount <= 0 or tx_type not in {"income", "expense"}:
            continue
        description = ""
        if has_desc:
            description = str(row.get(mapping.get("description")) or "").strip()
        description = " ".join(description.split())[:120] or "Transaction"
        rows.append(
            {
                "date": date_val,
                "description": description,
                "amount": float(amount),
                "type": tx_type,
                "category": "Other",
            }
        )
    return rows, True


def _read_csv(data: bytes) -> pd.DataFrame:
    for kwargs in (
        {"encoding": "utf-8-sig"},
        {"encoding": "utf-8"},
        {"encoding": "cp1252"},
        {"encoding": "latin1"},
    ):
        try:
            return pd.read_csv(io.BytesIO(data), nrows=MAX_ROWS, **kwargs)
        except Exception:
            continue
    raise HTTPException(status_code=422, detail="Could not read CSV file.")


def _read_xlsx(data: bytes) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(data), engine="openpyxl", nrows=MAX_ROWS)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Could not read Excel file.") from exc


def extract_from_tabular(data: bytes, source_type: str) -> Dict[str, Any]:
    warnings: List[str] = []
    df = _read_csv(data) if source_type == "csv" else _read_xlsx(data)
    mapped_rows, mapped = rows_from_dataframe(df)
    if mapped and mapped_rows:
        transactions = validate_transactions(mapped_rows, warnings)
        return {"transactions": transactions, "warnings": warnings, "used_llm": False}

    # Fallback to LLM if deterministic mapping fails.
    try:
        text = df.fillna("").to_csv(index=False)
    except Exception:
        text = df.astype(str).fillna("").to_csv(index=False)
    result = extract_transactions_with_llm(text, source_type)
    result.setdefault("warnings", [])
    if not mapped:
        result["warnings"].insert(0, "Could not confidently map columns; used LLM extraction.")
    elif not mapped_rows:
        result["warnings"].insert(0, "Deterministic mapping produced no valid rows; used LLM extraction.")
    return result


def _extract_pdf_text(data: bytes) -> str:
    try:
        import pdfplumber
    except Exception as exc:
        raise HTTPException(status_code=500, detail="PDF support is not available.") from exc

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) > MAX_PDF_PAGES:
                raise HTTPException(status_code=422, detail="PDF has too many pages. Maximum is 15 pages.")
            texts: List[str] = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    texts.append(page_text)
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                for table in tables:
                    for row in table or []:
                        if not row:
                            continue
                        cells = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
                        if cells:
                            texts.append(" | ".join(cells))
            combined = "\n".join(texts).strip()
            if not combined:
                raise HTTPException(
                    status_code=422,
                    detail="Scanned or password-protected PDF is not supported. Please upload a text-based statement or a CSV/Excel export.",
                )
            return combined
    except HTTPException:
        raise
    except Exception as exc:
        msg = str(exc).lower()
        if "password" in msg or "encrypted" in msg:
            raise HTTPException(
                status_code=422,
                detail="Scanned or password-protected PDF is not supported. Please upload a text-based statement or a CSV/Excel export.",
            ) from exc
        raise HTTPException(status_code=422, detail="Could not read PDF file.") from exc


def extract_from_pdf(data: bytes) -> Dict[str, Any]:
    text = _extract_pdf_text(data)
    result = extract_transactions_with_llm(text, "pdf")
    return result


def _ocr_image(data: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image, ImageOps
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Image OCR support is not available.") from exc

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        with Image.open(tmp_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("L")
            width, height = img.size
            if width < 1000 or height < 1000:
                scale = max(1000 / max(width, 1), 1000 / max(height, 1), 1.0)
                new_size = (int(width * scale), int(height * scale))
                resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                img = img.resize(new_size, resample)
            text = pytesseract.image_to_string(img)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Could not extract text from the image.")
    return text


def extract_from_image(data: bytes) -> Dict[str, Any]:
    text = _ocr_image(data)
    result = extract_transactions_with_llm(text, "image")
    # Receipt usually produces one expense transaction; cap handled in validation.
    return result


def parse_upload(data: bytes, source_type: str) -> Dict[str, Any]:
    if source_type in {"csv", "xlsx"}:
        return extract_from_tabular(data, source_type)
    if source_type == "pdf":
        return extract_from_pdf(data)
    if source_type in {"png", "jpg"}:
        result = extract_from_image(data)
        # Normalize source_type label to "image" in response handled by caller.
        return result
    raise HTTPException(status_code=415, detail="Unsupported file type.")
