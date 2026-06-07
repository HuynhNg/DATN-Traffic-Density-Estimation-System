from __future__ import annotations

import io
import zipfile
from collections import Counter, deque
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

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
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
        '<sheet name="Summary" sheetId="1" r:id="rId1"/>'
        '<sheet name="History" sheetId="2" r:id="rId2"/>'
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


def _unique_vehicle_counts(rows: list[dict[str, Any]], window_sec: int) -> list[int]:
    counts: list[int] = []
    active_counts: Counter[int] = Counter()
    active_rows: deque[tuple[float, set[int]]] = deque()

    for row in rows:
        current_ts = float(row.get("timestamp", 0.0))
        cutoff = current_ts - window_sec
        row_track_ids = _track_ids_from_row(row)

        active_rows.append((current_ts, row_track_ids))
        for track_id in row_track_ids:
            active_counts[track_id] += 1

        while active_rows and active_rows[0][0] < cutoff:
            _old_ts, old_track_ids = active_rows.popleft()
            for track_id in old_track_ids:
                active_counts[track_id] -= 1
                if active_counts[track_id] <= 0:
                    del active_counts[track_id]

        if active_counts:
            counts.append(len(active_counts))
        else:
            counts.append(int(row.get("objects_in_frame", 0)))
    return counts


def build_metrics_workbook(
    job_id: str,
    history: list[dict[str, Any]],
    selected_window: str,
) -> bytes:
    sorted_history = sorted(history, key=lambda row: float(row.get("timestamp", 0.0)))
    latest = sorted_history[-1] if sorted_history else {}
    selected_total = latest.get("total_vehicles", 0)

    summary_rows = [
        ["Metric", "Value"],
        ["Job ID", job_id],
        ["Selected Total Window", selected_window],
        ["Rows", len(sorted_history)],
        ["Latest Time", latest.get("time", "")],
        ["Latest Active Objects", latest.get("objects_in_frame", 0)],
        ["Latest Total Vehicles", selected_total],
        ["Latest Occupancy %", latest.get("occupancy_pct", 0)],
        ["Latest PCE Count", latest.get("pce_count", 0)],
        ["Latest Alert", latest.get("alert_label", "")],
    ]

    history_rows: list[list[Any]] = [
        [
            "timestamp",
            "time",
            "objects_in_frame",
            "total_vehicles_1min",
            "total_vehicles_1hour",
            "track_ids",
            "occupancy_pct",
            "pce_count",
            "alert_level",
            "alert_label",
            "fps",
        ]
    ]
    minute_totals = _unique_vehicle_counts(sorted_history, 60)
    hour_totals = _unique_vehicle_counts(sorted_history, 3600)
    for idx, row in enumerate(sorted_history):
        history_rows.append(
            [
                row.get("timestamp", 0),
                row.get("time", ""),
                row.get("objects_in_frame", 0),
                minute_totals[idx],
                hour_totals[idx],
                ",".join(str(track_id) for track_id in row.get("track_ids", [])),
                row.get("occupancy_pct", 0),
                row.get("pce_count", 0),
                row.get("alert_level", 0),
                row.get("alert_label", ""),
                row.get("fps", 0),
            ]
        )

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml())
        archive.writestr("_rels/.rels", _root_rels_xml())
        archive.writestr("xl/workbook.xml", _workbook_xml())
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml())
        archive.writestr("xl/worksheets/sheet1.xml", _sheet_xml(summary_rows))
        archive.writestr("xl/worksheets/sheet2.xml", _sheet_xml(history_rows))

    return output.getvalue()
