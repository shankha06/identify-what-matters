"""Loopback-only receiver for rendered Google snippets collected in the browser."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]


def load_state() -> tuple[dict, dict[str, dict], Path]:
    panel = json.loads((ROOT / "config" / "panel.json").read_text(encoding="utf-8"))
    output = ROOT / "output" / "collections" / f"google_instagram_mock_{datetime.now():%Y%m%d}.json"
    collection = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {
        "schema_version": 1,
        "source": "Google Search snippets; not Instagram API data",
        "search_query_template": "@{account_name} instagram",
        "methodology": "Google snippets only; comments are estimates when not shown in the snippet.",
        "collection_date": datetime.now().date().isoformat(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "accounts": [],
    }
    return panel, {row["username"]: row for row in collection.get("accounts", [])}, output


class Handler(BaseHTTPRequestHandler):
    def respond(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        panel, records, _ = load_state()
        accounts = panel.get("accounts", [])
        completed = {"latest_posts_panel", "organic_instagram_results",
                     "profile_only_or_no_post_snippets", "no_matching_results", "google_blocked"}
        if self.path == "/health":
            self.respond({"total": len(accounts), "saved": len(records)})
        elif self.path == "/accounts":
            self.respond({"total": len(accounts), "saved": len(records), "accounts": [
                row["username"] for row in accounts
                if records.get(row["username"], {}).get("search_status") not in completed
            ]})
        else:
            self.respond({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path != "/results":
            self.respond({"error": "not found"}, 404)
            return
        origin = self.headers.get("Origin", "")
        if origin and origin not in {"https://www.google.com", "https://google.com", "http://127.0.0.1:8766", "null"}:
            self.respond({"error": "unexpected origin"}, 403)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 2_000_000:
            self.respond({"error": "invalid payload size"}, 413)
            return
        try:
            form = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
            incoming = json.loads(form["payload"][0])
            if not isinstance(incoming, list):
                raise ValueError("payload must be a list")
            panel, records, output = load_state()
            configured = {row["username"]: row.get("group") for row in panel.get("accounts", [])}
            for record in incoming:
                username = record.get("username")
                if username not in configured:
                    raise ValueError(f"unknown account: {username}")
                record["group"] = configured[username]
                records[username] = record
            collection = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {
                "schema_version": 1,
                "source": "Google Search snippets; not Instagram API data",
                "collection_date": datetime.now().date().isoformat(),
                "accounts": [],
            }
            collection["search_query_template"] = "@{account_name} instagram"
            collection["accounts"] = [records[row["username"]] for row in panel["accounts"]
                                      if row["username"] in records]
            collection["updated_at"] = datetime.now(timezone.utc).isoformat()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(collection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.respond({"saved": len(incoming), "total_saved": len(records)})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.respond({"error": str(error)}, 400)

    def log_message(self, format: str, *args: object) -> None:
        print("[google receiver] " + format % args, flush=True)


if __name__ == "__main__":
    print("Listening on http://127.0.0.1:8766", flush=True)
    HTTPServer(("127.0.0.1", 8766), Handler).serve_forever()