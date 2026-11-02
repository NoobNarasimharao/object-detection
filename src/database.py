import json
import sqlite3
from pathlib import Path

class Database:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP, source TEXT, output TEXT, summary TEXT)')

    def save(self, run_id, source, output, summary):
        with sqlite3.connect(self.path) as db:
            db.execute('INSERT INTO runs(id,source,output,summary) VALUES(?,?,?,?)',
                       (run_id, str(source), str(output), json.dumps(summary)))

    def recent(self):
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute('SELECT * FROM runs ORDER BY created DESC LIMIT 100')]
