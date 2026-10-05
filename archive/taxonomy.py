"""
India Trend Mining Taxonomy & Discovery Engine
Provides curated taxonomy for discovering what is trending in India across:
1. Fashion & Apparel (Gen Z, Ethnic, Streetwear, Fast Fashion)
2. Entertainment, Memes & Bollywood
3. Tech, Startups & Creator Economy
4. Food & Regional Culture
"""

from typing import List, Dict

INDIA_FASHION_SEEDS = [
    "indiantrending", "delhifashion", "mumbaifashion", "streetwearindia",
    "chikankari", "lehengacholi", "sareelove", "kurtacollection",
    "genzfashion", "y2kindia", "thriftindia", "meeshohaul",
    "myntrafashion", "ajioluxe", "zaraoutfitsindia", "fastfashionindia",
    "festivewearindia", "diwalifit", "weddingseasonindia", "indowestern"
]

INDIA_POP_CULTURE_SEEDS = [
    "indianmemes", "reelsindia", "bollywoodfashion", "desiwedding",
    "desicomedy", "bangalorelife", "delhidiaries", "mumbailife",
    "indiatravel", "punjabisuits", "southindianfashion"
]

FAST_FASHION_ATTRIBUTES = {
    "aesthetic": ["y2k", "cottagecore", "minimalist", "clean girl", "streetwear", "coquette", "boho", "old money"],
    "silhouette": ["oversized", "crop top", "flared", "corset", "baggy", "wrap dress", "wide leg", "tiered"],
    "color_palette": ["pastel", "earthy", "rust", "electric blue", "sage green", "lavender", "emerald", "metallics"],
    "fabric_pattern": ["linen", "dola silk", "chikankari", "organza", "ribbed", "crochet", "velvet", "tie dye"],
    "intent_signals": ["where to buy", "link", "price", "dm please", "available", "dupe", "haul", "obsessed", "code"]
}
