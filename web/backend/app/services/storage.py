from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger("app.storage")


def _is_dangerous_storage_path(path: Path) -> bool:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    root = Path(resolved.anchor).resolve()

    return (
        resolved == root
        or resolved == cwd
        or resolved == cwd.parent
        or resolved == home
    )


def clean_storage_dir(save_dir: str) -> None:
    storage_path = Path(save_dir).resolve()
    if _is_dangerous_storage_path(storage_path):
        logger.warning("Skip storage cleanup for unsafe path: %s", storage_path)
        return

    storage_path.mkdir(parents=True, exist_ok=True)
    removed = 0
    for item in storage_path.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            removed += 1
        except OSError as exc:
            logger.warning("Failed to remove storage item %s: %s", item, exc)

    logger.info("Cleaned storage directory %s, removed %s item(s)", storage_path, removed)
