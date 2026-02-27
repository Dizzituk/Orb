# FILE: app/finance/services/drive_watcher_service.py
"""
Google Drive folder watcher for automatic statement imports.

Polls registered Drive folders for new PDF files,
downloads them, parses transactions, and imports to the DB.
Tracks processed files to avoid duplicate imports.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.finance.models import (
    CreditCard, CreditCardStatement, DriveWatchFolder,
    DriveProcessedFile, CreditCardTransaction,
)
from app.finance.services.drive_auth_service import get_drive_service
from app.finance.services.pdf_statement_parser import parse_statement_pdf
from app.finance.services.credit_card_service import import_parsed_transactions

logger = logging.getLogger(__name__)


@dataclass
class WatchScanResult:
    """Result of scanning a single watch folder."""
    card_name: str = ""
    folder_name: str = ""
    files_found: int = 0
    new_files: int = 0
    imported: list = field(default_factory=list)
    errors: list = field(default_factory=list)


@dataclass
class FullScanResult:
    """Result of scanning all watch folders."""
    folders_scanned: int = 0
    total_new_files: int = 0
    total_transactions: int = 0
    results: list[WatchScanResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def list_drive_folders(parent_folder_id: str) -> list[dict]:
    """List subfolders in a Google Drive folder.
    
    Used to discover card-specific subfolders in the
    main ASTRA Statements folder.
    """
    service = get_drive_service()
    if not service:
        return []

    try:
        results = service.files().list(
            q=f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(id, name, modifiedTime)",
            orderBy="name",
        ).execute()
        return results.get("files", [])
    except Exception as e:
        logger.error("[drive_watcher] List folders error: %s", e)
        return []


def list_pdfs_in_folder(folder_id: str) -> list[dict]:
    """List PDF files in a Google Drive folder."""
    service = get_drive_service()
    if not service:
        return []

    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
            fields="files(id, name, modifiedTime, size)",
            orderBy="modifiedTime desc",
        ).execute()
        return results.get("files", [])
    except Exception as e:
        logger.error("[drive_watcher] List PDFs error: %s", e)
        return []


def download_drive_file(file_id: str, dest_path: str) -> bool:
    """Download a file from Google Drive."""
    service = get_drive_service()
    if not service:
        return False

    try:
        from io import BytesIO
        from googleapiclient.http import MediaIoBaseDownload

        request = service.files().get_media(fileId=file_id)
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        with open(dest_path, "wb") as f:
            f.write(fh.getvalue())

        return True
    except Exception as e:
        logger.error("[drive_watcher] Download error for %s: %s", file_id, e)
        return False


def register_watch_folder(
    db: Session, card_id: int, drive_folder_id: str, folder_name: str = "",
) -> DriveWatchFolder:
    """Register a Drive folder to watch for a specific card."""
    existing = db.query(DriveWatchFolder).filter(
        DriveWatchFolder.card_id == card_id,
        DriveWatchFolder.drive_folder_id == drive_folder_id,
    ).first()

    if existing:
        existing.is_active = True
        existing.folder_name = folder_name or existing.folder_name
        db.commit()
        return existing

    watch = DriveWatchFolder(
        card_id=card_id,
        drive_folder_id=drive_folder_id,
        folder_name=folder_name,
    )
    db.add(watch)
    db.commit()
    db.refresh(watch)
    return watch


def scan_watch_folder(db: Session, watch: DriveWatchFolder) -> WatchScanResult:
    """Scan a single watch folder for new statements."""
    card = db.query(CreditCard).get(watch.card_id)
    result = WatchScanResult(
        card_name=card.name if card else "Unknown",
        folder_name=watch.folder_name,
    )

    # List PDFs in folder
    pdfs = list_pdfs_in_folder(watch.drive_folder_id)
    result.files_found = len(pdfs)

    for pdf_file in pdfs:
        file_id = pdf_file["id"]
        filename = pdf_file["name"]

        # Skip if already processed
        already = db.query(DriveProcessedFile).filter(
            DriveProcessedFile.drive_file_id == file_id
        ).first()
        if already:
            continue

        result.new_files += 1
        logger.info("[drive_watcher] New file: %s (%s)", filename, file_id)

        # Download to temp
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            if not download_drive_file(file_id, tmp_path):
                result.errors.append(f"Failed to download: {filename}")
                _record_processed(db, file_id, filename, watch.card_id, 0, "failed", "Download failed")
                continue

            # Parse PDF
            parse_result = parse_statement_pdf(tmp_path)

            if not parse_result.transactions:
                # Try AI vision as fallback
                try:
                    from app.finance.services.pdf_ai_parser import parse_pdf_with_vision
                    ai_result = parse_pdf_with_vision(tmp_path)
                    if ai_result.transactions:
                        from app.finance.services.pdf_statement_parser import ParsedTransaction
                        for tx in ai_result.transactions:
                            try:
                                d = datetime.strptime(tx["date"], "%d/%m/%Y").date()
                                parse_result.transactions.append(ParsedTransaction(
                                    transaction_date=d,
                                    description=tx["description"],
                                    amount=float(tx["amount"]),
                                    is_credit=tx.get("is_credit", False),
                                ))
                            except (ValueError, KeyError):
                                continue
                        parse_result.strategy_used = "ai_vision"
                except Exception as e:
                    logger.warning("[drive_watcher] AI fallback failed: %s", e)

            # Import transactions
            summary = import_parsed_transactions(db, watch.card_id, parse_result.transactions)

            # Create statement record
            stmt = CreditCardStatement(
                card_id=watch.card_id,
                statement_date=parse_result.statement_date or datetime.now().date(),
                opening_balance=parse_result.opening_balance or 0.0,
                closing_balance=parse_result.closing_balance or 0.0,
                minimum_payment=parse_result.minimum_payment,
                transactions_imported=summary.imported,
                source_filename=filename,
                drive_file_id=file_id,
                total_charges=summary.total_spend,
                total_payments=0.0,
            )
            db.add(stmt)
            db.flush()

            # Record processed file
            _record_processed(
                db, file_id, filename, watch.card_id,
                summary.imported, "success", None, stmt.id,
            )

            result.imported.append({
                "filename": filename,
                "transactions": summary.imported,
                "auto_categorised": summary.auto_categorised,
                "needs_review": summary.needs_review,
                "strategy": parse_result.strategy_used,
                "closing_balance": parse_result.closing_balance,
            })

        except Exception as e:
            logger.error("[drive_watcher] Error processing %s: %s", filename, e)
            result.errors.append(f"{filename}: {e}")
            _record_processed(db, file_id, filename, watch.card_id, 0, "failed", str(e))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Update last checked
    watch.last_checked = datetime.now()
    db.commit()
    return result


def scan_all_watch_folders(db: Session) -> FullScanResult:
    """Scan all active watch folders. Called by the scheduled job."""
    result = FullScanResult()

    watches = db.query(DriveWatchFolder).filter(
        DriveWatchFolder.is_active == True
    ).all()

    result.folders_scanned = len(watches)

    for watch in watches:
        try:
            scan_result = scan_watch_folder(db, watch)
            result.results.append(scan_result)
            result.total_new_files += scan_result.new_files
            for imp in scan_result.imported:
                result.total_transactions += imp["transactions"]
        except Exception as e:
            result.errors.append(f"Folder {watch.folder_name}: {e}")

    return result


def _record_processed(
    db: Session, file_id: str, filename: str, card_id: int,
    tx_count: int, status: str, error: str = None,
    statement_id: int = None,
):
    """Record a processed file to prevent re-imports."""
    record = DriveProcessedFile(
        drive_file_id=file_id,
        drive_filename=filename,
        card_id=card_id,
        statement_id=statement_id,
        transactions_imported=tx_count,
        status=status,
        error_message=error,
    )
    db.add(record)
