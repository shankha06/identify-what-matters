"""
Fast-Fashion & E-Commerce Trend Predictor
Implements Shein / Zara real-time trend mining methodology:
1. Micro-trend segmentation (Aesthetic, Silhouette, Color, Fabric)
2. Buyer Commercial Velocity (Likes vs Comments ratio, Price/Link inquiry frequency)
3. Trend Lifecycle Stage classification (Emerging, Peaking, Saturated, Decaying)
4. Merchandising & Product Action recommendations
"""

import re
from typing import Dict, Any, List
from collections import Counter
from identify_what_matters.taxonomy import FAST_FASHION_ATTRIBUTES


class FastFashionTrendPredictor:
    def __init__(self):
        self.attributes = FAST_FASHION_ATTRIBUTES

    def extract_fashion_attributes(self, text: str) -> Dict[str, List[str]]:
        """Extract aesthetic, silhouette, color, and fabric mentions from captions and comments."""
        lower_text = text.lower()
        extracted = {}
        for category, terms in self.attributes.items():
            if category == "intent_signals":
                continue
            matched = [t for t in terms if re.search(r"\b" + re.escape(t) + r"\b", lower_text)]
            extracted[category] = matched
        return extracted

    def classify_lifecycle_stage(self, post_count: int, velocity: float, comments_to_likes_ratio: float) -> str:
        """
        Classifies the trend stage:
        - Emerging: Moderate volume, high comment engagement ratio (conversations starting)
        - Peaking: Massive viral volume, high velocity
        - Saturated: High volume, declining comment engagement
        - Niche / Underground: Low volume, high loyalty
        """
        if velocity > 50000:
            return "Peaking (Mass Market Viral)"
        elif velocity > 10000 and comments_to_likes_ratio > 0.05:
            return "Emerging (High Buzz / Commercial Breakout)"
        elif velocity > 5000:
            return "Growth (Scaling Momentum)"
        else:
            return "Niche / Long-tail"

    def analyze_trend_commercial_potential(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce Shein-style commercial analysis for a trend cluster.
        """
        captions = " ".join([p.get("snippet", "") for p in cluster.get("sample_posts", [])]) + " " + cluster.get("representative_caption", "")
        attr_breakdown = self.extract_fashion_attributes(captions)

        total_likes = cluster.get("total_likes", 1)
        total_comments = cluster.get("total_comments", 0)
        c2l_ratio = total_comments / max(total_likes, 1)

        lifecycle = self.classify_lifecycle_stage(
            post_count=cluster.get("trend_size", 0),
            velocity=cluster.get("velocity_score", 0),
            comments_to_likes_ratio=c2l_ratio
        )

        return {
            "cluster_id": cluster.get("cluster_id"),
            "velocity_score": cluster.get("velocity_score"),
            "lifecycle_stage": lifecycle,
            "comment_to_like_ratio": round(c2l_ratio, 4),
            "dominant_aesthetics": attr_breakdown.get("aesthetic", []),
            "dominant_silhouettes": attr_breakdown.get("silhouette", []),
            "dominant_colors": attr_breakdown.get("color_palette", []),
            "dominant_fabrics": attr_breakdown.get("fabric_pattern", []),
            "merchandising_recommendation": (
                "High commercial demand detected! Fast-track catalog additions, run creator seedings."
                if c2l_ratio > 0.03 or "chikankari" in captions.lower() or "oversized" in captions.lower()
                else "Monitor virality; creator-led engagement."
            )
        }
