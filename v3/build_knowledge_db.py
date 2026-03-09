#!/usr/bin/env python3
import sqlite3
import os

DB_PATH = "data/v3/mirror_knowledge.db"

def build_db():
    # Ensure directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Remove existing if any
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tables
    cursor.execute("""
    CREATE TABLE knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT,
        query TEXT,
        content TEXT
    )
    """)
    
    # Data entries
    knowledge_entries = [
        # Self Knowledge
        ("self", "who created you", "I was created by Dipesh Majithia."),
        ("self", "creator", "My creator is Dipesh Majithia."),
        ("self", "who are you", "I am MirrorAI, a personal AI assistant designed to help you with tasks and information."),
        ("self", "what is your name", "My name is MirrorAI."),
        ("self", "who is dipesh majithia", "Dipesh Majithia is a software engineer and the creator of MirrorAI."),
        
        # Geography
        ("geo", "capital of india", "New Delhi is the capital of India."),
        ("geo", "capital of france", "Paris is the capital of France."),
        ("geo", "capital of japan", "Tokyo is the capital of Japan."),
        ("geo", "capital of usa", "Washington, D.C. is the capital of the United States."),
        ("geo", "highest mountain", "Mount Everest is the highest mountain on Earth, with a peak at 8,848 meters."),
        ("geo", "longest river", "The Nile is traditionally considered the longest river in the world, though some studies suggest the Amazon is longer."),
        ("geo", "largest country", "Russia is the largest country in the world by land area."),
        ("geo", "population of china", "China's population is approximately 1.4 billion people."),
        
        # Science
        ("science", "speed of light", "The speed of light in a vacuum is approximately 299,792,458 meters per second."),
        ("science", "photosynthesis", "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water."),
        ("science", "dna", "DNA, or deoxyribonucleic acid, is the molecule that carries genetic instructions in all living things."),
        ("science", "gravity", "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, it gives weight to objects."),
        ("science", "periodic table", "The periodic table is a tabular display of the chemical elements, organized by atomic number and chemical properties."),
        
        # History
        ("history", "industrial revolution", "The Industrial Revolution was a period of transition to new manufacturing processes in Europe and the US, occurring from about 1760 to 1840."),
        ("history", "world war 2", "World War II was a global conflict that lasted from 1939 to 1945, involving the vast majority of the world's countries."),
        ("history", "first man on moon", "Neil Armstrong became the first person to walk on the moon on July 20, 1969, during the Apollo 11 mission."),
        
        # Miscellaneous
        ("misc", "richest person", "The title of the world's richest person often changes between individuals like Elon Musk, Jeff Bezos, and Bernard Arnault."),
        ("misc", "premier league", "The Premier League is the top level of the English football league system."),
    ]
    
    # Insert
    cursor.executemany("INSERT INTO knowledge (category, query, content) VALUES (?, ?, ?)", knowledge_entries)
    conn.commit()
    conn.close()
    print(f"✅ Knowledge DB built with {len(knowledge_entries)} entries at {DB_PATH}")

if __name__ == "__main__":
    build_db()
