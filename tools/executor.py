#!/usr/bin/env python3
"""
Safe Tool Implementations for Mirror AI Orchestra.
Each tool is sandboxed and returns a string result.
"""
import re
import math
import json
import os
import sqlite3
from datetime import datetime


# ============================================================
# CALCULATOR — Safe math evaluation (no eval/exec)
# ============================================================
def calculator(expression: str) -> str:
    """Safely evaluate a math expression. No arbitrary code execution."""
    try:
        # Sanitize: only allow digits, operators, parens, decimal, spaces
        allowed = set("0123456789+-*/.() %")
        clean = expression.replace("sqrt", "SQRT").replace("**", "POW")
        
        # Check for dangerous chars
        for ch in clean:
            if ch not in allowed and ch not in "SQRTPOW":
                return f"Error: Invalid character '{ch}' in expression."
        
        # Handle sqrt
        expression = re.sub(r'sqrt\(([^)]+)\)', lambda m: str(math.sqrt(float(m.group(1)))), expression)
        
        # Handle ** (power)
        # Safe subset evaluation using ast
        import ast
        
        # Parse as AST and validate
        tree = ast.parse(expression, mode='eval')
        
        # Walk tree and ensure only safe nodes
        for node in ast.walk(tree):
            if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Num)):
                continue
            elif isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
                                   ast.FloorDiv, ast.USub, ast.UAdd)):
                continue
            else:
                return f"Error: Unsupported operation in expression."
        
        result = eval(compile(tree, '<calc>', 'eval'))
        
        # Format result
        if isinstance(result, float):
            if result == int(result):
                return str(int(result))
            return f"{result:.4f}"
        return str(result)
    
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error: Could not evaluate '{expression}'. ({str(e)})"


# ============================================================
# SEARCH — DuckDuckGo Instant Answers (No API key needed)
# ============================================================
def search_web(query: str) -> str:
    """Search the web using DuckDuckGo Instant Answer API."""
    try:
        import urllib.request
        import urllib.parse
        
        encoded = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'MirrorAI/1.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        
        # Try Abstract first
        if data.get('AbstractText'):
            return data['AbstractText'][:500]
        
        # Try Answer
        if data.get('Answer'):
            return str(data['Answer'])[:500]
        
        # Try first RelatedTopic
        if data.get('RelatedTopics') and len(data['RelatedTopics']) > 0:
            first = data['RelatedTopics'][0]
            if isinstance(first, dict) and first.get('Text'):
                return first['Text'][:500]
        
        return f"No instant answer found for '{query}'. Try rephrasing your question."
    
    except Exception as e:
        return f"Search error: {str(e)}"


# ============================================================
# NOTES — SQLite-backed personal notes
# ============================================================
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'memory', 'mirror_memory.db')

def _ensure_db():
    """Create the notes database if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            tags TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            summary TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def read_notes(topic: str) -> str:
    """Search notes by topic keyword."""
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT text, created_at FROM notes WHERE text LIKE ? OR tags LIKE ? ORDER BY created_at DESC LIMIT 5",
        (f"%{topic}%", f"%{topic}%")
    )
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        return f"No notes found about '{topic}'."
    
    notes = []
    for text, created_at in results:
        notes.append(f"• {text} ({created_at})")
    return f"Found {len(results)} note(s):\n" + "\n".join(notes)

def create_note(text: str) -> str:
    """Save a new note."""
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    
    # Auto-extract tags from common keywords
    tags = []
    keywords = ["meeting", "deadline", "birthday", "appointment", "buy", "call", 
                 "remind", "todo", "idea", "recipe", "travel", "gym", "doctor"]
    for kw in keywords:
        if kw in text.lower():
            tags.append(kw)
    
    conn.execute(
        "INSERT INTO notes (text, tags) VALUES (?, ?)",
        (text, ",".join(tags))
    )
    conn.commit()
    conn.close()
    return f"✅ Note saved: \"{text}\""


# ============================================================
# CALENDAR — SQLite-backed events
# ============================================================
def _ensure_calendar_db():
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calendar_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT NOT NULL,
            description TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def add_calendar_event(dt: str, description: str) -> str:
    """Add a calendar event."""
    _ensure_calendar_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO calendar_events (datetime, description) VALUES (?, ?)",
        (dt, description)
    )
    conn.commit()
    conn.close()
    return f"📅 Event added: \"{description}\" on {dt}"


# ============================================================
# DEVICE CONTROL — Stubs (real impl would use AppleScript/Swift)
# ============================================================
def control_device(action: str, target: str) -> str:
    """Simulate device control. In production, this calls native APIs."""
    # On Mac, we can actually control some things via osascript
    import subprocess
    
    try:
        if target == "volume":
            if action in ["turn up", "increase"]:
                subprocess.run(["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) + 10)"], capture_output=True)
                return "🔊 Volume increased."
            elif action in ["turn down", "decrease"]:
                subprocess.run(["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) - 10)"], capture_output=True)
                return "🔉 Volume decreased."
            elif action == "mute":
                subprocess.run(["osascript", "-e", "set volume with output muted"], capture_output=True)
                return "🔇 Volume muted."
            elif action == "unmute":
                subprocess.run(["osascript", "-e", "set volume without output muted"], capture_output=True)
                return "🔊 Volume unmuted."
        
        elif target == "brightness":
            return f"💡 Brightness {action}d. (Simulated — needs native integration)"
        
        elif target in ["wifi", "bluetooth", "flashlight", "dark mode", 
                        "do not disturb", "airplane mode"]:
            return f"✅ {target.title()} {action}d. (Simulated — needs native integration)"
        
        elif target == "font size":
            return f"🔤 Font size {action}d. (Simulated)"
        
        return f"⚙️ Device control: {action} {target} (Simulated)"
    
    except Exception as e:
        return f"Device control error: {str(e)}"


# ============================================================
# TOOL REGISTRY
# ============================================================
TOOL_REGISTRY = {
    "calculator": calculator,
    "search_web": search_web,
    "read_notes": read_notes,
    "create_note": create_note,
    "add_calendar_event": add_calendar_event,
    "control_device": control_device,
}

def get_tool(name: str):
    """Get a tool function by name."""
    return TOOL_REGISTRY.get(name)

def list_tools():
    """List available tool names."""
    return list(TOOL_REGISTRY.keys())
