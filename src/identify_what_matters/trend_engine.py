"""
Trend Intelligence & Clustering Engine
Powered by:
- all-mpnet-base-v2 (Dense semantic embeddings, 768-d)
- Agglomerative / Cosine Semantic Clustering
- Comment Intent & Buyer Velocity Scoring (Shein-like demand signals)
- Automatic Topic Extraction & Trend Titling
"""

import os
import re
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

DEFAULT_LOCAL_MPNET_PATH = "/Users/s0r0qw6/model-py-java/artifacts/all-mpnet-base-v2"


class TrendIntelligenceEngine:
    def __init__(self, model_path: Optional[str] = None):
        target_path = model_path or (
            DEFAULT_LOCAL_MPNET_PATH if os.path.exists(DEFAULT_LOCAL_MPNET_PATH)
            else "sentence-transformers/all-mpnet-base-v2"
        )
        print(f"[TrendEngine] Initializing MPNet model from: {target_path}")
        self.model = SentenceTransformer(target_path)

        # High-intent buyer regex patterns (Shein / Zara product demand detection)
        self.intent_patterns = [
            r"\b(price|cost|how much|kitne ka|kitna)\b",
            r"\b(link|link please|dm link|link pls)\b",
            r"\b(available|in stock|where to buy|kahan se)\b",
            r"\b(dupe|affordable|code|coupon)\b",
            r"\b(loved|need this|obsessed|want this|must have)\b"
        ]

    def clean_text(self, text: str) -> str:
        """Strip URLs, excessive hashtags, and special symbols for cleaner embedding."""
        if not text:
            return ""
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"#[\w]+|@[\w.]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def compute_buyer_intent_score(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze comment section for purchase intent velocity.
        Calculates:
        - purchase_intent_ratio: % of comments asking for price/link/availability
        - high_intent_count: absolute number of potential buyer queries
        - sample_intent_quotes: top buyer questions
        """
        if not comments:
            return {"purchase_intent_ratio": 0.0, "high_intent_count": 0, "sample_intent_quotes": []}

        high_intent_comments = []
        for c in comments:
            raw_text = c.get("text", "").lower()
            if any(re.search(pat, raw_text) for pat in self.intent_patterns):
                high_intent_comments.append(c.get("text"))

        ratio = round(len(high_intent_comments) / len(comments), 3) if comments else 0.0
        return {
            "purchase_intent_ratio": ratio,
            "high_intent_count": len(high_intent_comments),
            "sample_intent_quotes": high_intent_comments[:5]
        }

    def cluster_posts_into_trends(
        self,
        posts: List[Dict[str, Any]],
        distance_threshold: float = 0.45,
        min_cluster_size: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Group posts into emerging trend topics using MPNet embeddings + Agglomerative Clustering.
        """
        if len(posts) < min_cluster_size:
            return []

        # Prepare texts
        cleaned_captions = [self.clean_text(p.get("caption", "")) for p in posts]
        valid_indices = [i for i, text in enumerate(cleaned_captions) if len(text) > 10]

        if len(valid_indices) < min_cluster_size:
            # Fallback if too few long texts: use all
            valid_indices = list(range(len(posts)))

        valid_posts = [posts[i] for i in valid_indices]
        valid_texts = [cleaned_captions[i] if len(cleaned_captions[i]) > 3 else posts[i].get("tag", "trend") for i in valid_indices]

        # 1. Encode with MPNet (768-dim semantic vectors)
        embeddings = self.model.encode(valid_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)

        # 2. Agglomerative Clustering with Cosine Distance
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average"
        )
        labels = clustering.fit_predict(embeddings)

        # 3. Organize Clusters
        clusters_map: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(labels):
            clusters_map.setdefault(lbl, []).append(idx)

        trend_results = []
        for lbl, member_indices in clusters_map.items():
            if len(member_indices) < min_cluster_size:
                continue
            cluster_posts = [valid_posts[idx] for idx in member_indices]
            cluster_embs = embeddings[member_indices]

            # Compute centroid & exemplar
            centroid = np.mean(cluster_embs, axis=0, keepdims=True)
            centroid = centroid / np.linalg.norm(centroid)
            sims = cosine_similarity(cluster_embs, centroid).flatten()
            exemplar_idx = member_indices[int(np.argmax(sims))]
            exemplar_post = valid_posts[exemplar_idx]

            # Aggregate engagement
            total_likes = sum(p.get("like_count", 0) for p in cluster_posts)
            total_comments = sum(p.get("comment_count", 0) for p in cluster_posts)
            avg_likes = total_likes / len(cluster_posts)

            # Extract common hashtags/keywords
            all_hashtags = []
            for p in cluster_posts:
                matches = re.findall(r"#(\w+)", p.get("caption", ""))
                all_hashtags.extend([m.lower() for m in matches])
            from collections import Counter
            top_hashtags = [tag for tag, cnt in Counter(all_hashtags).most_common(5)]

            # Velocity Score: (Total Likes + Total Comments * 5) * Size Penalty / Recency Factor
            velocity_score = round(total_likes + total_comments * 4.5, 2)

            trend_results.append({
                "cluster_id": int(lbl),
                "trend_size": len(cluster_posts),
                "total_likes": total_likes,
                "total_comments": total_comments,
                "velocity_score": velocity_score,
                "score_kind": "lifetime_engagement_not_velocity",
                "trend_status": "unverified_topic_candidate",
                "representative_caption": exemplar_post.get("caption", "")[:180],
                "top_hashtags": top_hashtags,
                "sample_posts": [
                    {
                        "id": p.get("id"),
                        "code": p.get("code"),
                        "username": p.get("username"),
                        "likes": p.get("like_count"),
                        "comments": p.get("comment_count"),
                        "snippet": p.get("caption", "")[:120]
                    }
                    for p in cluster_posts[:3]
                ]
            })

        # Sort trends by velocity_score descending
        trend_results.sort(key=lambda x: x["velocity_score"], reverse=True)
        return trend_results
