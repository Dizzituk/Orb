# FILE: app/finance/services/drive_folder_sync.py
"""
Unified Drive folder sync for credit card statements.

Links a Drive folder per credit card, auto-imports all PDFs,
tracks processed files to avoid duplicates.
New files dropped in the folder are picked up on next sync.

Same pattern as drive_screenshot_sync but for PDF statements.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from app.db import Base

logger = logging.getLogger(__name__)


class ProcessedStatement(Base):
    """Tracks which Drive statement files have been processed."""
    __tablename__ = "finance_processed_statements"

    id = Column(Integer, primary_key=True)
    drive_file_id = Column(String(200), unique=True, nullable=False, index=True)
    drive_filename = Column(String(500))
    card_id = Column(Integer, nullable=False, index=True)
    folder_id = Column(String(200))
    processed_at = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=False)
    transactions_imported = Column(Integer, default=0)
    error_message = Column(String(1000), nullable=True)


class CardFolderLink(Base):
    """Links a credit card to a Drive folder for auto-sync."""
    __tablename__ = "finance_card_folder_links"

    id = Column(Integer, primary_key=True)
    card_id = Column(Integer, nullable=False, unique=True, index=True)
    folder_id = Column(String(200), nullable=False)
    folder_name = Column(String(500))
    linked_at = Column(DateTime, default=datetime.utcnow)
    last_sync = Column(DateTime, nullable=True)
    total_synced = Column(Integer, default=0)


async def get_card_folder(db: Session, card_id: int) -> Optional[dict]:
    """Get the linked folder for a credit card."""
    link = db.query(CardFolderLink).filter(
        CardFolderLink.card_id == card_id
    ).first()
    if not link:
        return None
    return {
        "card_id": link.card_id,
        "folder_id": link.folder_id,
        "folder_name": link.folder_name,
        "linked_at": str(link.linked_at),
        "last_sync": str(link.last_sync) if link.last_sync else None,
        "total_synced": link.total_synced,
    }


async def link_card_folder(
    db: Session, card_id: int, folder_id: str, folder_name: str
) -> dict:
    """Link a Drive folder to a credit card for auto-sync."""
    existing = db.query(CardFolderLink).filter(
        CardFolderLink.card_id == card_id
    ).first()
    if existing:
        existing.folder_id = folder_id
        existing.folder_name = folder_name
        existing.linked_at = datetime.utcnow()
    else:
        existing = CardFolderLink(
            card_id=card_id,
            folder_id=folder_id,
            folder_name=folder_name,
        )
        db.add(existing)
    db.commit()
    logger.info("[folder_sync] Linked card %d to folder %s (%s)", card_id, folder_name, folder_id)
    return {"linked": True, "card_id": card_id, "folder_name": folder_name}


async def unlink_card_folder(db: Session, card_id: int) -> dict:
    """Remove the folder link for a credit card."""
    count = db.query(CardFolderLink).filter(
        CardFolderLink.card_id == card_id
    ).delete()
    db.commit()
    return {"unlinked": True, "removed": count}


async def sync_card_statements(db: Session, card_id: int) -> dict:
    """Scan linked folder and import all new statement PDFs.

    Downloads each unprocessed PDF, saves to temp file, parses it,
    and imports transactions via import_parsed_transactions.
    """
    import tempfile
    import os
    from app.finance.services.drive_browser_service import (
        list_folder_contents, download_file_bytes,
    )
    from app.finance.services.pdf_statement_parser import parse_statement_pdf
    from app.finance.services.credit_card_service import import_parsed_transactions

    link = db.query(CardFolderLink).filter(
        CardFolderLink.card_id == card_id
    ).first()
    if not link:
        return {"error": "No folder linked for this card."}

    # List PDFs in folder
    try:
        contents = list_folder_contents(link.folder_id, file_types=["pdf"])
    except Exception as e:
        logger.error("[folder_sync] Failed to list folder: %s", e)
        return {"error": f"Could not access Drive folder: {e}"}

    files = contents.get("files", [])
    if not files:
        return {
            "folder": link.folder_name,
            "found": 0,
            "new": 0,
            "imported": 0,
            "errors": 0,
            "results": [],
        }

    # Filter out already-processed files
    processed_ids = {
        r[0] for r in
        db.query(ProcessedStatement.drive_file_id).filter(
            ProcessedStatement.card_id == card_id
        ).all()
    }
    new_files = [f for f in files if f["id"] not in processed_ids]

    results = []
    error_count = 0
    total_tx = 0

    for file_info in new_files:
        file_id = file_info["id"]
        filename = file_info["name"]

        record = ProcessedStatement(
            drive_file_id=file_id,
            drive_filename=filename,
            card_id=card_id,
            folder_id=link.folder_id,
        )

        tmp_path = None
        try:
            file_bytes = download_file_bytes(file_id)
            if not file_bytes:
                record.error_message = "Empty file"
                record.success = False
                db.add(record)
                db.commit()
                error_count += 1
                continue

            # Save to temp file — parse_statement_pdf expects a file path
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False
            ) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            # Parse statement (returns StatementParseResult)
            parse_result = parse_statement_pdf(tmp_path)
            txs = parse_result.transactions

            if not txs:
                msg = "No transactions found"
                if parse_result.warnings:
                    msg += f" ({'; '.join(parse_result.warnings)})"
                record.error_message = msg
                record.success = False
                db.add(record)
                db.commit()
                error_count += 1
                results.append({
                    "filename": filename,
                    "success": False,
                    "message": msg,
                })
                continue

            # Import transactions (returns CCImportSummary)
            summary = import_parsed_transactions(db, card_id, txs)

            record.success = True
            record.transactions_imported = summary.imported
            total_tx += summary.imported

            period = ""
            if parse_result.statement_date:
                period = str(parse_result.statement_date)

            results.append({
                "filename": filename,
                "success": True,
                "transactions": summary.imported,
                "duplicates": summary.duplicates,
                "period": period,
            })

        except Exception as e:
            logger.error("[folder_sync] Failed %s: %s", filename, e)
            record.error_message = str(e)[:1000]
            record.success = False
            error_count += 1
            results.append({
                "filename": filename,
                "success": False,
                "message": str(e),
            })
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        db.add(record)
        db.commit()

    # Update sync timestamp
    link.last_sync = datetime.utcnow()
    link.total_synced += len(new_files)
    db.commit()

    return {
        "folder": link.folder_name,
        "found": len(files),
        "already_processed": len(processed_ids),
        "new": len(new_files),
        "imported": total_tx,
        "errors": error_count,
        "results": results,
    }
