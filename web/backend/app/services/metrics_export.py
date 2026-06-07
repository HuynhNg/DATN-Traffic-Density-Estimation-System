from __future__ import annotations

import io
import time
import zipfile
from collections import defaultdict
from typing import Any
from xml.sax.saxutils import escape


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _cell_xml(row_idx: int, col_idx: int, value: Any) -> str:
    ref = f"{_column_name(col_idx)}{row_idx}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _sheet_xml(rows: list[list[Any]]) -> str:
    sheet_rows: list[str] = []
    for row_idx, row in enumerate(rows, start=1):
        cells = "".join(_cell_xml(row_idx, col_idx, value) for col_idx, value in enumerate(row, start=1))
        sheet_rows.append(f'<row r="{row_idx}">{cells}</row>')

    column_widths = [18, 14, 32, 32, 12, 12, 12, 12]
    cols = "".join(
        f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>'
        for idx, width in enumerate(column_widths, start=1)
    )

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<cols>"
        + cols
        + "</cols>"
        '<sheetData>'
        + "".join(sheet_rows)
        + "</sheetData></worksheet>"
    )


def _workbook_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        '<sheet name="Theo phút" sheetId="1" r:id="rId1"/>'
        '<sheet name="Theo giờ" sheetId="2" r:id="rId2"/>'
        "</sheets></workbook>"
    )


def _workbook_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet2.xml"/>'
        "</Relationships>"
    )


def _root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def _content_types_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/worksheets/sheet2.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )


def _track_ids_from_row(row: dict[str, Any]) -> set[int]:
    track_ids = row.get("track_ids")
    if not isinstance(track_ids, list):
        return set()
    return {int(track_id) for track_id in track_ids if track_id is not None}


def _track_classes_from_row(row: dict[str, Any]) -> dict[int, str]:
    track_classes = row.get("track_classes")
    if not isinstance(track_classes, list):
        return {}

    result: dict[int, str] = {}
    for item in track_classes:
        if not isinstance(item, dict) or item.get("track_id") is None:
            continue
        result[int(item["track_id"])] = str(item.get("class_name", "")).lower()
    return result


def _bucket_label(bucket_start: int, bucket_seconds: int) -> str:
    fmt = "%Y-%m-%d %H:00" if bucket_seconds == 3600 else "%Y-%m-%d %H:%M"
    return time.strftime(fmt, time.localtime(bucket_start))


def _new_bucket() -> dict[str, Any]:
    return {
        "track_ids": set(),
        "track_classes": {},
        "right_to_left": 0,
        "left_to_right": 0,
    }


def _bucket_rows(
    history: list[dict[str, Any]],
    events: list[dict[str, Any]],
    bucket_seconds: int,
) -> list[list[Any]]:
    buckets: defaultdict[int, dict[str, Any]] = defaultdict(_new_bucket)

    for row in history:
        row_ts = float(row.get("timestamp", 0.0))
        bucket_start = int(row_ts // bucket_seconds) * bucket_seconds
        bucket = buckets[bucket_start]
        track_ids = _track_ids_from_row(row)
        bucket["track_ids"].update(track_ids)
        bucket["track_classes"].update(_track_classes_from_row(row))

    for event in events:
        event_ts = float(event.get("timestamp", 0.0))
        bucket_start = int(event_ts // bucket_seconds) * bucket_seconds
        bucket = buckets[bucket_start]
        event_type = event.get("type")
        if event_type == "right_to_left":
            bucket["right_to_left"] += 1
        elif event_type == "left_to_right":
            bucket["left_to_right"] += 1
        if event.get("track_id") is not None:
            track_id = int(event["track_id"])
            bucket["track_ids"].add(track_id)
            if event.get("class_name"):
                bucket["track_classes"][track_id] = str(event["class_name"]).lower()

    rows: list[list[Any]] = [
        [
            "Thời gian",
            "Tổng số xe",
            "Xe đi vào (phải sang trái)",
            "Xe đi ra (trái sang phải)",
            "Xe máy",
            "Ô tô",
            "Xe buýt",
            "Xe tải",
        ]
    ]

    for bucket_start in sorted(buckets):
        bucket = buckets[bucket_start]
        track_ids: set[int] = bucket["track_ids"]
        track_classes: dict[int, str] = bucket["track_classes"]
        class_counts = {"motor": 0, "car": 0, "bus": 0, "truck": 0}
        for track_id in track_ids:
            class_name = track_classes.get(track_id, "")
            if class_name in ("motor", "motorcycle"):
                class_counts["motor"] += 1
            elif class_name in class_counts:
                class_counts[class_name] += 1

        rows.append(
            [
                _bucket_label(bucket_start, bucket_seconds),
                len(track_ids),
                bucket["right_to_left"],
                bucket["left_to_right"],
                class_counts["motor"],
                class_counts["car"],
                class_counts["bus"],
                class_counts["truck"],
            ]
        )

    return rows


def build_metrics_workbook(
    job_id: str,
    history: list[dict[str, Any]],
    selected_window: str,
    flow_events: list[dict[str, Any]] | None = None,
) -> bytes:
    sorted_history = sorted(history, key=lambda row: float(row.get("timestamp", 0.0)))
    sorted_events = sorted(flow_events or [], key=lambda event: float(event.get("timestamp", 0.0)))
    _ = job_id, selected_window
    minute_rows = _bucket_rows(sorted_history, sorted_events, 60)
    hour_rows = _bucket_rows(sorted_history, sorted_events, 3600)

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml())
        archive.writestr("_rels/.rels", _root_rels_xml())
        archive.writestr("xl/workbook.xml", _workbook_xml())
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml())
        archive.writestr("xl/worksheets/sheet1.xml", _sheet_xml(minute_rows))
        archive.writestr("xl/worksheets/sheet2.xml", _sheet_xml(hour_rows))

    return output.getvalue()
