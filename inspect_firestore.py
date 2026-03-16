"""
inspect_firestore.py
--------------------
View all Firestore data from your terminal.

Usage:
    python inspect_firestore.py                  # show everything
    python inspect_firestore.py threads          # agent_threads only
    python inspect_firestore.py brands           # brands only
    python inspect_firestore.py brand_context    # brand_context chunks
    python inspect_firestore.py images           # images
    python inspect_firestore.py docs             # docs
    python inspect_firestore.py threads manar    # filter by user_id
"""

import sys
import json
import os
from dotenv import load_dotenv
from google.cloud import firestore

load_dotenv()

# ── Init ──────────────────────────────────────────────────────────────────────
PROJECT_ID = "creativo-bf5c8"
db = firestore.Client(project=PROJECT_ID, database="agent-orchestration")

# ── Helpers ───────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
DIM    = "\033[2m"

def header(title):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

def field(key, value, indent=2):
    pad = " " * indent
    if isinstance(value, (dict, list)):
        print(f"{pad}{YELLOW}{key}{RESET}: {json.dumps(value, ensure_ascii=False, indent=2).replace(chr(10), chr(10)+pad+'  ')}")
    else:
        print(f"{pad}{YELLOW}{key}{RESET}: {value}")

def truncate(text, limit=120):
    text = str(text)
    return text[:limit] + f"{DIM}…{RESET}" if len(text) > limit else text

# ── Collection printers ───────────────────────────────────────────────────────

def show_threads(user_filter=None):
    header("agent_threads")
    docs = db.collection("agent_threads").stream()
    count = 0
    for doc in docs:
        d = doc.to_dict()
        if user_filter and d.get("user_id") != user_filter:
            continue
        count += 1
        status_color = GREEN if d.get("status") == "active" else DIM
        print(f"\n  {BOLD}{doc.id}{RESET}  [{status_color}{d.get('status','?')}{RESET}]")
        print(f"    user: {d.get('user_id')}  |  agent: {d.get('agent_name')}  |  last_active: {d.get('last_active','?')}")

        fields = d.get("collected_fields", {})
        if fields:
            print(f"    {GREEN}collected_fields ({len(fields)}):{RESET}")
            for k, v in fields.items():
                print(f"      {YELLOW}{k}{RESET}: {truncate(v)}")

        messages = d.get("messages", [])
        if messages:
            print(f"    {GREEN}messages ({len(messages)} turns):{RESET}")
            for i, m in enumerate(messages[-4:]):  # last 4 messages
                role_color = CYAN if m["role"] == "user" else DIM
                print(f"      {role_color}[{m['role']}]{RESET} {truncate(m['content'], 100)}")
            if len(messages) > 4:
                print(f"      {DIM}... and {len(messages)-4} earlier messages{RESET}")
    print(f"\n  {DIM}Total: {count} thread(s){RESET}")


def show_brands(brand_filter=None):
    header("brands")
    docs = db.collection("brands").stream()
    count = 0
    for doc in docs:
        if brand_filter and doc.id != brand_filter:
            continue
        count += 1
        d = doc.to_dict()
        print(f"\n  {BOLD}{doc.id}{RESET}")
        for section, value in d.items():
            if isinstance(value, dict):
                non_empty = {k: v for k, v in value.items() if v}
                if non_empty:
                    print(f"    {GREEN}{section}{RESET}:")
                    for k, v in non_empty.items():
                        print(f"      {YELLOW}{k}{RESET}: {truncate(str(v))}")
            else:
                if value:
                    print(f"    {YELLOW}{section}{RESET}: {truncate(str(value))}")
    print(f"\n  {DIM}Total: {count} brand(s){RESET}")


def show_brand_context():
    header("brand_context")
    docs = db.collection("brand_context").stream()
    count = 0
    for doc in docs:
        count += 1
        d = doc.to_dict()
        print(f"\n  {BOLD}{doc.id}{RESET}")
        print(f"    brand_id: {d.get('brand_id')}  |  version: {d.get('version')}  |  chunk: {d.get('chunk_index')}")
        content = d.get("content", "")
        print(f"    content: {truncate(content, 200)}")
    print(f"\n  {DIM}Total: {count} chunk(s){RESET}")


def show_images(user_filter=None):
    header("images")
    docs = db.collection("images").stream()
    count = 0
    for doc in docs:
        d = doc.to_dict()
        if user_filter and d.get("user_id") != user_filter:
            continue
        count += 1
        status = d.get("status", "?")
        status_color = GREEN if status == "approved" else (RED if status == "rejected" else YELLOW)
        print(f"\n  {BOLD}{doc.id}{RESET}  [{status_color}{status}{RESET}]")
        print(f"    title: {d.get('title','?')}  |  platform: {d.get('platform','?')}  |  created: {d.get('created_at','?')}")
        if d.get("theme"):     print(f"    theme: {d.get('theme')}")
        if d.get("priority"):  print(f"    priority: {d.get('priority')}")
        if d.get("campaign"):  print(f"    campaign: {d.get('campaign')}")
    print(f"\n  {DIM}Total: {count} image(s){RESET}")


def show_docs():
    header("docs")
    docs_col = db.collection("docs").stream()
    count = 0
    for doc in docs_col:
        count += 1
        d = doc.to_dict()
        print(f"\n  {BOLD}{doc.id}{RESET}")
        print(f"    title: {d.get('title','?')}  |  category: {d.get('category','?')}")
        content = d.get("content", d.get("text", ""))
        if content:
            print(f"    content: {truncate(str(content), 150)}")
    print(f"\n  {DIM}Total: {count} doc(s){RESET}")


# ── Entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "threads":       show_threads,
    "brands":        show_brands,
    "brand_context": show_brand_context,
    "images":        show_images,
    "docs":          show_docs,
}

def main():
    args = sys.argv[1:]
    user_filter = args[1] if len(args) > 1 else None

    if not args or args[0] == "all":
        show_threads(user_filter)
        show_brands()
        show_brand_context()
        show_images(user_filter)
        show_docs()
    elif args[0] in COMMANDS:
        fn = COMMANDS[args[0]]
        # pass user_filter only to functions that accept it
        try:
            fn(user_filter) if user_filter else fn()
        except TypeError:
            fn()
    else:
        print(f"Unknown collection '{args[0]}'. Options: {', '.join(COMMANDS)} or 'all'")

if __name__ == "__main__":
    main()
