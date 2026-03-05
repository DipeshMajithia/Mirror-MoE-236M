#!/usr/bin/env python3
"""
Mirror AI — Local Memory System (Private RAG)

Provides:
1. SQLite-backed note/conversation storage
2. TF-IDF based lightweight embedding (no external model needed)
3. Semantic search via cosine similarity
4. Daily auto-summarization
"""
import os
import re
import math
import json
import sqlite3
from datetime import datetime, date
from collections import Counter

DB_PATH = os.path.join(os.path.dirname(__file__), "mirror_memory.db")

# ============================================================
# DATABASE LAYER
# ============================================================
def get_db():
    """Get a database connection, creating tables if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            tags TEXT DEFAULT '',
            embedding TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            session_id TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS daily_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            summary TEXT NOT NULL,
            key_topics TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS calendar_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT NOT NULL,
            description TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    return conn

# ============================================================
# LIGHTWEIGHT EMBEDDING (TF-IDF)
# No external model dependency — works instantly
# ============================================================
STOP_WORDS = set("the a an is was are were be been being have has had do does did "
                 "will would shall should may might can could of in to for on with "
                 "at by from as into through during before after above below between "
                 "and but or nor not so yet both either neither each every all any few "
                 "more most other some such no only own same than too very just don't "
                 "i me my we our you your he him his she her it its they them their "
                 "what which who whom this that these those am is are was were".split())

def tokenize_text(text: str) -> list:
    """Simple tokenizer: lowercase, split, remove stop words."""
    words = re.findall(r'[a-z0-9]+', text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]

def compute_tfidf(tokens: list, corpus_freq: dict = None, corpus_size: int = 1) -> dict:
    """Compute TF-IDF vector for a token list."""
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    
    vector = {}
    for word, count in tf.items():
        tf_val = count / total
        # IDF: if no corpus, use 1.0 (all terms equally important)
        if corpus_freq and word in corpus_freq:
            idf_val = math.log(corpus_size / (1 + corpus_freq[word]))
        else:
            idf_val = 1.0
        vector[word] = tf_val * idf_val
    
    return vector

def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Cosine similarity between two sparse vectors."""
    if not vec_a or not vec_b:
        return 0.0
    
    # Dot product
    common_keys = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
    
    # Magnitudes
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return dot / (mag_a * mag_b)

def embed_text(text: str) -> str:
    """Create a JSON-serialized TF-IDF embedding of text."""
    tokens = tokenize_text(text)
    vector = compute_tfidf(tokens)
    return json.dumps(vector)

# ============================================================
# MEMORY OPERATIONS
# ============================================================
def save_note(text: str, tags: str = "") -> str:
    """Save a note with auto-embedding."""
    conn = get_db()
    embedding = embed_text(text)
    
    # Auto-extract tags
    if not tags:
        keywords = ["meeting", "deadline", "birthday", "appointment", "buy", "call",
                     "remind", "todo", "idea", "recipe", "travel", "gym", "doctor",
                     "project", "code", "bug", "fix", "design", "password"]
        found = [kw for kw in keywords if kw in text.lower()]
        tags = ",".join(found)
    
    conn.execute(
        "INSERT INTO notes (text, tags, embedding) VALUES (?, ?, ?)",
        (text, tags, embedding)
    )
    conn.commit()
    conn.close()
    return f"✅ Note saved: \"{text}\""

def search_notes(query: str, top_k: int = 5) -> list:
    """Search notes using TF-IDF cosine similarity."""
    conn = get_db()
    cursor = conn.execute("SELECT id, text, tags, embedding, created_at FROM notes")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []
    
    query_vec = compute_tfidf(tokenize_text(query))
    
    scored = []
    for row in rows:
        note_id, text, tags, emb_json, created_at = row
        if emb_json:
            try:
                note_vec = json.loads(emb_json)
            except:
                note_vec = compute_tfidf(tokenize_text(text))
        else:
            note_vec = compute_tfidf(tokenize_text(text))
        
        sim = cosine_similarity(query_vec, note_vec)
        
        # Boost if query words appear in tags
        query_tokens = set(tokenize_text(query))
        tag_tokens = set(tags.lower().split(",")) if tags else set()
        if query_tokens & tag_tokens:
            sim += 0.3
        
        scored.append((sim, note_id, text, created_at))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]

def get_recent_notes(limit: int = 10) -> list:
    """Get most recent notes."""
    conn = get_db()
    cursor = conn.execute(
        "SELECT text, tags, created_at FROM notes ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    results = cursor.fetchall()
    conn.close()
    return results

# ============================================================
# CONVERSATION LOGGING
# ============================================================
def log_conversation(role: str, content: str, session_id: str = ""):
    """Log a conversation turn."""
    conn = get_db()
    conn.execute(
        "INSERT INTO conversations (role, content, session_id) VALUES (?, ?, ?)",
        (role, content, session_id)
    )
    conn.commit()
    conn.close()

def get_conversation_history(session_id: str = "", limit: int = 20) -> list:
    """Get recent conversation history."""
    conn = get_db()
    if session_id:
        cursor = conn.execute(
            "SELECT role, content, created_at FROM conversations WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit)
        )
    else:
        cursor = conn.execute(
            "SELECT role, content, created_at FROM conversations ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
    results = cursor.fetchall()
    conn.close()
    return list(reversed(results))

# ============================================================
# DAILY SUMMARY
# ============================================================
def generate_daily_summary() -> str:
    """Generate a summary of today's conversations and notes."""
    today = date.today().isoformat()
    
    conn = get_db()
    
    # Get today's conversations
    cursor = conn.execute(
        "SELECT role, content FROM conversations WHERE DATE(created_at) = ?",
        (today,)
    )
    convos = cursor.fetchall()
    
    # Get today's notes
    cursor = conn.execute(
        "SELECT text FROM notes WHERE DATE(created_at) = ?",
        (today,)
    )
    notes = cursor.fetchall()
    conn.close()
    
    if not convos and not notes:
        return "No activity today."
    
    # Build summary
    summary_parts = []
    
    if convos:
        user_msgs = [c[1] for c in convos if c[0] == "user"]
        summary_parts.append(f"Conversations: {len(convos)} messages ({len(user_msgs)} from user)")
        
        # Extract key topics
        all_text = " ".join([c[1] for c in convos])
        tokens = tokenize_text(all_text)
        top_words = Counter(tokens).most_common(5)
        topics = [w[0] for w in top_words]
        summary_parts.append(f"Key topics: {', '.join(topics)}")
    
    if notes:
        summary_parts.append(f"Notes created: {len(notes)}")
        for n in notes[:3]:
            summary_parts.append(f"  • {n[0][:80]}")
    
    summary = "\n".join(summary_parts)
    
    # Save the summary
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO daily_summaries (date, summary, key_topics) VALUES (?, ?, ?)",
        (today, summary, ",".join(topics) if convos else "")
    )
    conn.commit()
    conn.close()
    
    return summary

def get_memory_context(query: str, max_items: int = 3) -> str:
    """Build a memory context string for injection into prompts."""
    results = search_notes(query, top_k=max_items)
    
    if not results:
        return ""
    
    context_parts = ["Relevant memories:"]
    for sim, note_id, text, created_at in results:
        if sim > 0.1:  # Only include if somewhat relevant
            context_parts.append(f"• {text} ({created_at})")
    
    if len(context_parts) == 1:
        return ""
    
    return "\n".join(context_parts)


# ============================================================
# CALENDAR OPERATIONS
# ============================================================
def add_event(dt: str, description: str) -> str:
    """Add a calendar event."""
    conn = get_db()
    conn.execute(
        "INSERT INTO calendar_events (datetime, description) VALUES (?, ?)",
        (dt, description)
    )
    conn.commit()
    conn.close()
    return f"📅 Event added: \"{description}\" on {dt}"

def get_upcoming_events(limit: int = 5) -> list:
    """Get upcoming calendar events."""
    conn = get_db()
    cursor = conn.execute(
        "SELECT datetime, description FROM calendar_events WHERE datetime >= ? ORDER BY datetime LIMIT ?",
        (datetime.now().isoformat(), limit)
    )
    results = cursor.fetchall()
    conn.close()
    return results
