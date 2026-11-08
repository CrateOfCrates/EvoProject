import os
import sys
import time
import argparse
import logging
import csv
import re
from typing import Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Complements for fast nucleotide operations
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def revcomp(seq: str) -> str:
    return seq.translate(COMP)[::-1]

def comp(b: str) -> str:
    return b.translate(COMP)

# Cached reference sequence
_FULL_REF_SEQ: Optional[str] = None

def get_tp53_ref_seq() -> str:
    global _FULL_REF_SEQ
    if _FULL_REF_SEQ is None:
        ref_path = os.path.join(os.path.dirname(__file__), "data", "full.txt")
        if os.path.exists(ref_path):
            with open(ref_path, "r", encoding="utf-8") as f:
                _FULL_REF_SEQ = f.read().replace("\n", "").strip()
        else:
            raise FileNotFoundError(f"Reference file not found at {ref_path}")
    return _FULL_REF_SEQ

def c_pos_to_g(c_str: str) -> int:
    """
    Map cDNA position string (e.g. '376', '376-1', '559+2') of TP53 (NM_000546.6)
    to GRCh38 genomic coordinate on NC_000017.11 (minus strand).
    """
    m = re.match(r"^(-?\d+)([+-]\d+)?$", c_str)
    if not m:
        raise ValueError(f"Invalid cDNA coordinate format: {c_str}")
    base_c = int(m.group(1))
    offset = int(m.group(2)) if m.group(2) else 0

    if 376 <= base_c <= 559:
        g = 7675236 - (base_c - 376)
    elif 560 <= base_c <= 672:
        g = 7674971 - (base_c - 560)
    elif 673 <= base_c <= 782:
        g = 7674290 - (base_c - 673)
    elif 783 <= base_c <= 919:
        g = 7673837 - (base_c - 783)
    else:
        raise ValueError(f"Position c.{base_c} out of mapped TP53 exons 5-8 range (376-919)")

    # TP53 is on minus strand: + offset on c is - offset on g; - offset on c is + offset on g
    return g - offset

def get_ref_bases(g_start: int, g_end: int) -> str:
    """
    Fetch reference bases on NC_000017.11 forward strand from g_start to g_end (inclusive).
    """
    seq = get_tp53_ref_seq()
    # In data/full.txt, start coordinate is 7687490 (minus strand transcript sense sequence)
    # Forward strand base at coordinate x is comp(seq[7687490 - x])
    return "".join(comp(seq[7687490 - x]) for x in range(g_start, g_end + 1))

def hgvs_to_spdi_fast(hgvs_str: str) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str], Optional[str]]:
    """
    Convert an HGVS transcript variant (e.g. NM_000546.6:c.376_377del) to
    genomic SPDI components: (seq_id, pos_0_based, deleted_seq, inserted_seq, spdi_str).
    High-throughput local conversion without network latency.
    """
    if not isinstance(hgvs_str, str) or not hgvs_str.strip() or hgvs_str == "NA" or ":" not in hgvs_str:
        return (None, None, None, None, None)

    seq_id = "NC_000017.11"
    c_part = hgvs_str.split(":", 1)[1]
    if c_part.startswith("c."):
        c_part = c_part[2:]

    # 1. Delins
    if "delins" in c_part:
        pos_part, ins = c_part.split("delins")
        if "_" in pos_part:
            p1, p2 = pos_part.split("_")
            g1, g2 = c_pos_to_g(p1), c_pos_to_g(p2)
        else:
            g1 = g2 = c_pos_to_g(pos_part)
        g_start, g_end = min(g1, g2), max(g1, g2)
        pos_0 = g_start - 1
        del_seq = get_ref_bases(g_start, g_end)
        ins_seq = revcomp(ins)

    # 2. Deletion
    elif "del" in c_part:
        pos_part = c_part.replace("del", "")
        if "_" in pos_part:
            p1, p2 = pos_part.split("_")
            g1, g2 = c_pos_to_g(p1), c_pos_to_g(p2)
        else:
            g1 = g2 = c_pos_to_g(pos_part)
        g_start, g_end = min(g1, g2), max(g1, g2)
        pos_0 = g_start - 1
        del_seq = get_ref_bases(g_start, g_end)
        ins_seq = ""

    # 3. Insertion
    elif "ins" in c_part:
        pos_part, ins = c_part.split("ins")
        p1, p2 = pos_part.split("_")
        g1, g2 = c_pos_to_g(p1), c_pos_to_g(p2)
        g_start, g_end = min(g1, g2), max(g1, g2)
        pos_0 = g_start - 1
        del_seq = ""
        ins_seq = revcomp(ins)

    # 4. Duplication
    elif "dup" in c_part:
        pos_part = c_part.replace("dup", "")
        if "_" in pos_part:
            p1, p2 = pos_part.split("_")
            g1, g2 = c_pos_to_g(p1), c_pos_to_g(p2)
        else:
            g1 = g2 = c_pos_to_g(pos_part)
        g_start, g_end = min(g1, g2), max(g1, g2)
        pos_0 = g_end
        del_seq = ""
        ins_seq = get_ref_bases(g_start, g_end)

    # 5. Substitution (SNV) e.g. 919G>A
    elif ">" in c_part:
        pos_part, change = c_part.rsplit(">", 1)
        ref_base = pos_part[-1]
        pos_only = pos_part[:-1]
        alt_base = change
        g = c_pos_to_g(pos_only)
        pos_0 = g - 1
        del_seq = get_ref_bases(g, g)
        ins_seq = revcomp(alt_base)

    else:
        raise ValueError(f"Unsupported variant type: {hgvs_str}")

    spdi_str = f"{seq_id}:{pos_0}:{del_seq}:{ins_seq}"
    return (seq_id, pos_0, del_seq, ins_seq, spdi_str)


def hgvs_to_spdi(hgvs_str: str, assembly: str = "GRCh38") -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    """
    Standard function signature for backward compatibility.
    Returns: (seq_id, pos_0_based, deleted_seq, inserted_seq).
    """
    seq_id, pos_0, del_seq, ins_seq, _ = hgvs_to_spdi_fast(hgvs_str)
    return (seq_id, pos_0, del_seq, ins_seq)


def main():
    parser = argparse.ArgumentParser(
        description="Convert HGVS transcript variants to genomic SPDI coordinates at maximum throughput."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="urn_mavedb_00001277-a-1_custom.csv",
        help="Input CSV file path (default: urn_mavedb_00001277-a-1_custom.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="urn_mavedb_00001277-a-1_spdi.csv",
        help="Output CSV file path (default: urn_mavedb_00001277-a-1_spdi.csv)",
    )
    parser.add_argument(
        "--hgvs-col",
        type=str,
        default="hgvs_nt",
        help="Column name containing HGVS variants (default: hgvs_nt)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on rows to process",
    )

    args = parser.parse_args()

    logger.info(f"Loading input file: {args.input}")
    start_time = time.time()

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if args.limit:
        rows = rows[:args.limit]

    total_rows = len(rows)
    logger.info(f"Converting {total_rows} variants from column '{args.hgvs_col}'...")

    # Extra SPDI fields to append
    spdi_fields = ["spdi_seq_id", "spdi_pos_0_based", "spdi_deleted_seq", "spdi_inserted_seq", "spdi"]
    output_fields = fieldnames + [f for f in spdi_fields if f not in fieldnames]

    converted_count = 0
    failed_count = 0

    for idx, row in enumerate(rows):
        hgvs_val = row.get(args.hgvs_col, "")
        try:
            seq_id, pos_0, del_seq, ins_seq, spdi_str = hgvs_to_spdi_fast(hgvs_val)
            row["spdi_seq_id"] = seq_id if seq_id is not None else ""
            row["spdi_pos_0_based"] = pos_0 if pos_0 is not None else ""
            row["spdi_deleted_seq"] = del_seq if del_seq is not None else ""
            row["spdi_inserted_seq"] = ins_seq if ins_seq is not None else ""
            row["spdi"] = spdi_str if spdi_str is not None else ""
            converted_count += 1
        except Exception as e:
            logger.warning(f"Error converting row {idx + 1} ('{hgvs_val}'): {e}")
            for field in spdi_fields:
                row[field] = ""
            failed_count += 1

        if (idx + 1) % 2000 == 0 or (idx + 1) == total_rows:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"Progress: {idx + 1}/{total_rows} ({rate:.1f} variants/sec)")

    logger.info(f"Writing fully converted dataset to {args.output}...")
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    elapsed_total = time.time() - start_time
    logger.info(
        f"Conversion complete! Converted: {converted_count}, Failed: {failed_count} in {elapsed_total:.2f}s "
        f"({total_rows / elapsed_total:.1f} variants/sec). Output saved to: {args.output}"
    )


if __name__ == "__main__":
    main()
