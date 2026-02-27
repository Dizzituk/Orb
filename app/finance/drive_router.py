# FILE: app/finance/drive_router.py
"""
Google Drive integration endpoints.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/finance/drive",
    tags=["Finance - Drive"],
    dependencies=[Depends(require_auth)],
)

# ─── Google Drive Integration ────────────────────────────

@router.get("/status")
async def drive_status():
    """Check Google Drive authentication status."""
    from app.finance.services.drive_auth_service import check_auth_status
    return check_auth_status()


@router.post("/auth")
async def drive_auth():
    """Start Google Drive OAuth2 authentication flow.
    Opens a browser window for the user to log in to Google.
    """
    from app.finance.services.drive_auth_service import start_auth_with_local_server
    return start_auth_with_local_server()


@router.post("/revoke")
async def drive_revoke():
    """Revoke Google Drive access."""
    from app.finance.services.drive_auth_service import revoke_auth
    return revoke_auth()


@router.get("/folders")
async def drive_list_folders(parent_id: str):
    """List subfolders in a Google Drive folder."""
    from app.finance.services.drive_watcher_service import list_drive_folders
    return list_drive_folders(parent_id)


@router.get("/folder-pdfs")
async def drive_list_pdfs(folder_id: str):
    """List PDF files in a Google Drive folder."""
    from app.finance.services.drive_watcher_service import list_pdfs_in_folder
    return list_pdfs_in_folder(folder_id)


@router.post("/watch")
async def drive_register_watch(data: dict, db: Session = Depends(get_db)):
    """Register a Drive folder to watch for a credit card."""
    from app.finance.services.drive_watcher_service import register_watch_folder
    watch = register_watch_folder(
        db, data["card_id"], data["drive_folder_id"],
        folder_name=data.get("folder_name", ""),
    )
    return {
        "id": watch.id, "card_id": watch.card_id,
        "folder_name": watch.folder_name, "drive_folder_id": watch.drive_folder_id,
    }


@router.get("/watches")
async def drive_list_watches(db: Session = Depends(get_db)):
    """List all active watch folder registrations."""
    from app.finance.models import DriveWatchFolder, CreditCard
    watches = db.query(DriveWatchFolder).filter(DriveWatchFolder.is_active == True).all()
    result = []
    for w in watches:
        card = db.query(CreditCard).get(w.card_id)
        result.append({
            "id": w.id, "card_id": w.card_id, "card_name": card.name if card else "?",
            "folder_name": w.folder_name, "drive_folder_id": w.drive_folder_id,
            "last_checked": str(w.last_checked) if w.last_checked else None,
        })
    return result


@router.post("/scan")
async def drive_scan_now(db: Session = Depends(get_db)):
    """Manually trigger a scan of all watch folders."""
    from app.finance.services.drive_watcher_service import scan_all_watch_folders
    result = scan_all_watch_folders(db)
    return {
        "folders_scanned": result.folders_scanned,
        "new_files": result.total_new_files,
        "transactions_imported": result.total_transactions,
        "results": [
            {
                "card": r.card_name,
                "folder": r.folder_name,
                "new_files": r.new_files,
                "imported": r.imported,
                "errors": r.errors,
            }
            for r in result.results
        ],
        "errors": result.errors,
    }


@router.post("/scan/{card_id}")
async def drive_scan_card(card_id: int, db: Session = Depends(get_db)):
    """Scan watch folders for a specific card."""
    from app.finance.services.drive_watcher_service import scan_watch_folder
    from app.finance.models import DriveWatchFolder
    watches = db.query(DriveWatchFolder).filter(
        DriveWatchFolder.card_id == card_id, DriveWatchFolder.is_active == True
    ).all()
    results = []
    for watch in watches:
        r = scan_watch_folder(db, watch)
        results.append({
            "card": r.card_name, "new_files": r.new_files,
            "imported": r.imported, "errors": r.errors,
        })
    return {"results": results}


@router.get("/scheduler")
async def drive_scheduler_status():
    """Get Drive folder scanner scheduler status."""
    from app.finance.services.drive_scheduler import get_scheduler_status
    return get_scheduler_status()


@router.post("/scheduler/start")
async def drive_scheduler_start(data: dict = None):
    """Start the Drive folder polling scheduler."""
    from app.finance.services.drive_scheduler import start_drive_scheduler
    interval = (data or {}).get("poll_minutes", 30)
    start_drive_scheduler(interval)
    return {"started": True, "poll_minutes": interval}


@router.post("/scheduler/stop")
async def drive_scheduler_stop():
    """Stop the Drive folder polling scheduler."""
    from app.finance.services.drive_scheduler import stop_drive_scheduler
    stop_drive_scheduler()
    return {"stopped": True}


@router.get("/processed")
async def drive_processed_files(card_id: Optional[int] = None, db: Session = Depends(get_db)):
    """List processed Drive files."""
    from app.finance.models import DriveProcessedFile
    q = db.query(DriveProcessedFile)
    if card_id:
        q = q.filter(DriveProcessedFile.card_id == card_id)
    files = q.order_by(DriveProcessedFile.processed_at.desc()).limit(50).all()
    return [
        {
            "id": f.id, "filename": f.drive_filename, "card_id": f.card_id,
            "transactions": f.transactions_imported, "status": f.status,
            "processed_at": str(f.processed_at), "error": f.error_message,
        }
        for f in files
    ]




# ─── Drive File Browser ────────────────────────────────

@router.get("/browse/root")
async def drive_browse_root():
    """List top-level folders in Google Drive."""
    from app.finance.services.drive_browser_service import list_root_folders
    return list_root_folders()


@router.get("/browse/{folder_id}")
async def drive_browse_folder(folder_id: str, types: str = ""):
    """List contents of a Drive folder. ?types=pdf,csv to filter."""
    from app.finance.services.drive_browser_service import list_folder_contents
    file_types = [t.strip() for t in types.split(",") if t.strip()] if types else None
    return list_folder_contents(folder_id, file_types)


@router.get("/download/{file_id}")
async def drive_download_file(file_id: str):
    """Download a file from Drive and return its bytes as base64."""
    import base64
    from app.finance.services.drive_browser_service import download_file_bytes
    file_bytes = download_file_bytes(file_id)
    return {
        "size": len(file_bytes),
        "data_b64": base64.b64encode(file_bytes).decode(),
    }


@router.get("/search")
async def drive_search_files(q: str, types: str = ""):
    """Search Drive for files matching query. ?types=pdf,csv to filter."""
    from app.finance.services.drive_browser_service import search_files
    file_types = [t.strip() for t in types.split(",") if t.strip()] if types else None
    return search_files(q, file_types)


@router.post("/import-to-van")
async def drive_import_to_van(data: dict, db: Session = Depends(get_db)):
    """Import a van finance agreement PDF from Google Drive.
    
    Body: {"file_id": "...", "filename": "..."}
    """
    import base64
    from app.finance.services.drive_browser_service import download_file_bytes
    from app.finance.services.van_pdf_parser import parse_van_finance_pdf
    from app.finance.services.van_finance_service import (
        auto_populate_van_from_transactions,
        create_van_finance,
    )

    file_id = data.get("file_id")
    if not file_id:
        return {"error": "file_id required"}

    # Download from Drive
    file_bytes = download_file_bytes(file_id)

    # Parse with OCR
    pdf_data = await parse_van_finance_pdf(file_bytes, data.get("filename", "agreement.pdf"))

    # Cross-reference transactions
    discovered = auto_populate_van_from_transactions(db)
    mb = discovered.get("moneybarn", {})
    dvla = discovered.get("dvla", {})

    van_data = {
        "vehicle_description": pdf_data.get("vehicle_description") or f"Van ({pdf_data.get('registration', 'Unknown')})",
        "purchase_price": pdf_data.get("purchase_price") or 0,
        "deposit_paid": pdf_data.get("deposit_paid") or 0,
        "finance_amount": pdf_data.get("finance_amount") or 0,
        "apr": pdf_data.get("apr") or 0,
        "monthly_payment": mb.get("monthly_payment") or pdf_data.get("monthly_payment") or 0,
        "total_payments": pdf_data.get("total_payments") or 48,
        "payments_made": mb.get("payments_made", 0),
        "first_payment_date": mb.get("first_payment_date") or pdf_data.get("agreement_date"),
        "finance_provider": pdf_data.get("finance_provider") or "Moneybarn",
        "business_use_percentage": 100,
        "road_tax_amount": dvla.get("annual_amount"),
        "cost_method": "mileage",
    }

    van = create_van_finance(db, van_data)

    return {
        "created": True,
        "id": van.id,
        "extracted": pdf_data,
        "transactions_found": {
            "moneybarn_payments": mb.get("payments_made", 0),
            "dvla_payments": dvla.get("payments_found", 0),
        },
    }


@router.post("/import-statement")
async def drive_import_statement(data: dict, db: Session = Depends(get_db)):
    """Import a credit card statement PDF from Google Drive.
    
    Body: {"file_id": "...", "card_id": ..., "filename": "..."}
    """
    from app.finance.services.drive_browser_service import download_file_bytes
    from app.finance.services.pdf_statement_parser import parse_statement_pdf
    from app.finance.services.credit_card_service import import_card_transactions

    file_id = data.get("file_id")
    card_id = data.get("card_id")
    if not file_id or not card_id:
        return {"error": "file_id and card_id required"}

    file_bytes = download_file_bytes(file_id)

    # Parse the statement
    parse_result = parse_statement_pdf(file_bytes, data.get("filename", "statement.pdf"))
    if parse_result.get("error"):
        return {"error": parse_result["error"], "can_retry_ai": True}

    # Import transactions
    result = import_card_transactions(
        db, card_id,
        parse_result.get("transactions", []),
        parse_result.get("metadata", {}),
    )

    return result




# ─── Credit Card Folder Sync ─────────────────────────────

@router.get("/cards/{card_id}/folder")
async def get_card_folder(card_id: int, db: Session = Depends(get_db)):
    """Get linked Drive folder for a credit card."""
    from app.finance.services.drive_folder_sync import get_card_folder
    config = await get_card_folder(db, card_id)
    if not config:
        return {"linked": False}
    return {"linked": True, **config}


@router.post("/cards/{card_id}/link-folder")
async def link_card_folder(card_id: int, body: dict, db: Session = Depends(get_db)):
    """Link a Drive folder to a credit card for auto-import."""
    from app.finance.services.drive_folder_sync import link_card_folder
    folder_id = body.get("folder_id")
    folder_name = body.get("folder_name", "Statements")
    if not folder_id:
        return {"error": "folder_id required"}
    return await link_card_folder(db, card_id, folder_id, folder_name)


@router.post("/cards/{card_id}/unlink-folder")
async def unlink_card_folder(card_id: int, db: Session = Depends(get_db)):
    """Remove folder link for a credit card."""
    from app.finance.services.drive_folder_sync import unlink_card_folder
    return await unlink_card_folder(db, card_id)


@router.post("/cards/{card_id}/sync")
async def sync_card_folder(card_id: int, db: Session = Depends(get_db)):
    """Sync all new statement PDFs from the linked Drive folder."""
    from app.finance.services.drive_folder_sync import sync_card_statements
    return await sync_card_statements(db, card_id)


# ─── Screenshot Folder Sync ──────────────────────────────

@router.get("/screenshots/folder")
async def get_screenshot_folder(db: Session = Depends(get_db)):
    """Get the linked screenshot Drive folder config."""
    from app.finance.services.drive_screenshot_sync import get_linked_folder
    config = await get_linked_folder(db)
    if not config:
        return {"linked": False}
    return {"linked": True, **config}


@router.post("/screenshots/link-folder")
async def link_screenshot_folder(
    body: dict,
    db: Session = Depends(get_db),
):
    """Link a Drive folder for screenshot auto-sync."""
    from app.finance.services.drive_screenshot_sync import link_folder
    folder_id = body.get("folder_id")
    folder_name = body.get("folder_name", "Screenshots")
    if not folder_id:
        return {"error": "folder_id is required"}
    return await link_folder(db, folder_id, folder_name)


@router.post("/screenshots/unlink")
async def unlink_screenshot_folder(db: Session = Depends(get_db)):
    """Remove the linked screenshot folder."""
    from app.finance.services.drive_screenshot_sync import unlink_folder
    return await unlink_folder(db)


@router.post("/screenshots/sync")
async def sync_screenshots(db: Session = Depends(get_db)):
    """Scan linked folder for new screenshots and process them."""
    from app.finance.services.drive_screenshot_sync import sync_screenshots
    return await sync_screenshots(db)


@router.get("/screenshots/history")
async def screenshot_sync_history(
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """Get recent screenshot processing history."""
    from app.finance.services.drive_screenshot_sync import get_sync_history
    return await get_sync_history(db, limit)

