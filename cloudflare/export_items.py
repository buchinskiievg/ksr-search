"""
Dump local sqlite items table into chunked SQL files for D1 import.

Why chunks: `wrangler d1 execute --file` works but a single 200 MB SQL file
will time out / OOM. Splitting into ~5000-row chunks keeps each upload under
a minute.

Output: cloudflare/migrations/0002_items_*.sql (overwrites)
"""
import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
SQLITE_PATH = os.path.join(os.path.dirname(HERE), "ksr.sqlite")
OUT_DIR = os.path.join(HERE, "migrations")

CHUNK_ROWS = 5000


def esc(s: str | None) -> str:
    if s is None:
        return "NULL"
    return "'" + str(s).replace("'", "''") + "'"


def main():
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM items")
    total = cur.fetchone()[0]
    print(f"Total items: {total:,}")

    # Wipe old chunks
    for fn in os.listdir(OUT_DIR):
        if fn.startswith("0002_items_") and fn.endswith(".sql"):
            os.remove(os.path.join(OUT_DIR, fn))

    cur.execute("SELECT id, sheet, code, name, unit, category FROM items ORDER BY id")
    chunk_idx = 0
    written = 0
    buf = []
    for r in cur:
        vals = f"({r[0]},{esc(r[1])},{esc(r[2])},{esc(r[3])},{esc(r[4])},{esc(r[5])})"
        buf.append(vals)
        if len(buf) >= CHUNK_ROWS:
            chunk_idx += 1
            fn = os.path.join(OUT_DIR, f"0002_items_{chunk_idx:03d}.sql")
            with open(fn, "w", encoding="utf-8") as f:
                f.write("INSERT INTO items(id,sheet,code,name,unit,category) VALUES\n")
                f.write(",\n".join(buf) + ";\n")
            written += len(buf)
            print(f"  chunk {chunk_idx}: {written:,}/{total:,}")
            buf = []
    if buf:
        chunk_idx += 1
        fn = os.path.join(OUT_DIR, f"0002_items_{chunk_idx:03d}.sql")
        with open(fn, "w", encoding="utf-8") as f:
            f.write("INSERT INTO items(id,sheet,code,name,unit,category) VALUES\n")
            f.write(",\n".join(buf) + ";\n")
        written += len(buf)
        print(f"  chunk {chunk_idx}: {written:,}/{total:,}")
    conn.close()
    print(f"Done. {chunk_idx} chunks in {OUT_DIR}")


if __name__ == "__main__":
    main()
