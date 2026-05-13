#!/usr/bin/env bash
# Daily backup of the production DuckDB.
# Cron-friendly: idempotent, non-interactive, exits non-zero on missing source.
#
# Limitation: uses plain `cp`. DuckDB's WAL keeps the .duckdb file in a mostly
# consistent state but a torn write during cp is theoretically possible. If
# corrupted backups are ever observed, upgrade to DuckDB's online `.backup`
# syntax via a small Python wrapper.
#
# Cron entry (recommended — 23:00 daily, after market close at 15:00):
#     0 23 * * * /opt/stock_price_v3/scripts/backup_db.sh >> /var/log/quant-v6-backup.log 2>&1
#
# Restore: stop the bot, then
#     cp backups/quant_v6_core_YYYYMMDD.duckdb data/quant_v6_core.duckdb

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$PROJECT_ROOT/data/quant_v6_core.duckdb"
DST_DIR="$PROJECT_ROOT/backups"
TS="$(date +%Y%m%d)"
DST="$DST_DIR/quant_v6_core_${TS}.duckdb"

mkdir -p "$DST_DIR"

if [[ ! -f "$SRC" ]]; then
    echo "[backup_db] ERROR: source not found at $SRC" >&2
    exit 1
fi

cp -- "$SRC" "$DST"

# Cross-platform file size (GNU stat -c, BSD stat -f).
SIZE="$(stat -c%s "$DST" 2>/dev/null || stat -f%z "$DST" 2>/dev/null || echo '?')"
echo "[backup_db] $(date +%Y-%m-%dT%H:%M:%S) backed up: $DST ($SIZE bytes)"

# Retention: keep the last 14 days. `-mtime +14` deletes files modified
# more than 14 days ago, so we retain ~15 backups including today's.
find "$DST_DIR" -maxdepth 1 -name 'quant_v6_core_*.duckdb' -type f -mtime +14 -print -delete
