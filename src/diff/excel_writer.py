import time
import logging
from pathlib import Path
import pandas as pd
from openpyxl.styles import Font, PatternFill

logger = logging.getLogger("excel_writer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def write_excel(records: list[dict], output_path: str, sheet_name: str = "Data", retries: int = 3):
    """
    Writes a list of dicts to a styled .xlsx file.
    Safe against Windows file-lock errors (e.g. file open in Excel) via
    write-to-temp + atomic rename, with retry-with-backoff.
    """
    output_path = Path(output_path)
    temp_path = output_path.with_suffix(".tmp.xlsx")

    df = pd.DataFrame(records)

    # Write to a temp file first
    with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]

        # Style the header row
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        for cell in worksheet[1]:
            cell.font = header_font
            cell.fill = header_fill

        # Auto-fit column widths (rough estimate based on content length)
        for column_cells in worksheet.columns:
            max_length = max(len(str(cell.value)) for cell in column_cells)
            col_letter = column_cells[0].column_letter
            worksheet.column_dimensions[col_letter].width = max_length + 4

    # Atomically move temp -> real target, retrying if the real file is locked (open in Excel)
    delay = 2
    for attempt in range(1, retries + 1):
        try:
            temp_path.replace(output_path)
            logger.info(f"Saved: {output_path}")
            return
        except PermissionError:
            logger.warning(
                f"'{output_path.name}' is locked (probably open in Excel). "
                f"Retry {attempt}/{retries} in {delay}s..."
            )
            time.sleep(delay)
            delay *= 2

    raise PermissionError(
        f"Could not save '{output_path}' after {retries} attempts — "
        f"please close it if it's open in Excel and try again."
    )


if __name__ == "__main__":
    sample_records = [
        {"title": "A Light in the Attic", "price": "£51.77", "availability": "In stock", "rating": "Three"},
        {"title": "Tipping the Velvet", "price": "£53.74", "availability": "In stock", "rating": "One"},
    ]
    write_excel(sample_records, "output.xlsx")