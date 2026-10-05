"""Conservative, content-level political discourse and coordination audit."""

from collections import Counter, defaultdict
import re


PARTIES = {
    "BJP": re.compile(r"\b(?:bjp|bharatiya janata party)\b", re.I),
    "Congress": re.compile(r"\b(?:congress party|inc party|indian national congress|congress)\b", re.I),
    "AAP": re.compile(r"\baam aadmi party\b", re.I),
    "TMC": re.compile(r"\b(?:tmc|trinamool congress)\b", re.I),
    "SP": re.compile(r"\b(?:samajwadi party)\b", re.I),
}
FAVOR = re.compile(r"\b(?:i support|we support|vote for|voting for|backing|stand with|supporting|good work by|well done)\b", re.I)
OPPOSE = re.compile(r"\b(?:vote against|never vote for|stop voting for|oppose|against|resign|corrupt|failed|shame on|boycott)\b", re.I)
NEGATED = re.compile(r"\b(?:not|never|don't|do not|no)\b", re.I)


def party_position(text: str) -> tuple[str, str] | None:
    """Only classify a single explicitly named party with an unambiguous position."""
    matches = [party for party, pattern in PARTIES.items() if pattern.search(text)]
    if len(matches) != 1 or "?" in text or len(text) > 300:
        return None
    cleaned = text.strip()
    favor = bool(FAVOR.search(re.sub(r"\b(?:never vote for|stop voting for|vote against)\b", "", cleaned, flags=re.I)))
    oppose = bool(OPPOSE.search(cleaned))
    if favor == oppose:
        return None
    if NEGATED.search(cleaned) and not re.search(r"\b(?:never vote for|stop voting for)\b", cleaned, re.I):
        return None
    party_match = PARTIES[matches[0]].search(cleaned)
    cue = FAVOR.search(cleaned) if favor else OPPOSE.search(cleaned)
    if not party_match or not cue or abs(party_match.start() - cue.start()) > 70:
        return None
    return matches[0], "favor" if favor else "oppose"


def audit_comments(comments: list[dict]) -> dict:
    """Report content-level evidence; no account-level political labels or bot verdicts."""
    party_counts: dict[str, Counter] = defaultdict(Counter)
    seen = set()
    by_template: dict[str, set[str]] = defaultdict(set)
    with_ids = 0
    valid = 0
    for comment in comments:
        text = str(comment.get("text") or "").strip()
        if not text:
            continue
        valid += 1
        account = str(comment.get("author_id") or (comment.get("user") or {}).get("pk") or "")
        if account:
            with_ids += 1
            normalized = " ".join(re.findall(r"\w+", text.casefold()))
            if len(normalized) >= 35:
                by_template[normalized].add(account)
        position = party_position(text)
        if position:
            key = (account or f"row:{valid}", position)
            if key not in seen:
                seen.add(key)
                party_counts[position[0]][position[1]] += 1
    templates = list(by_template.items())
    groups = []
    for text, accounts in templates[:500]:
        shingles = {text[i:i + 5] for i in range(len(text) - 4)}
        for group in groups:
            other = group["shingles"]
            if len(shingles & other) / max(len(shingles | other), 1) >= 0.85:
                group["accounts"].update(accounts)
                break
        else:
            groups.append({"shingles": shingles, "accounts": set(accounts)})
    repeated = [group for group in groups if len(group["accounts"]) >= 3]
    clusters = [len(group["accounts"]) for group in repeated]
    return {
        "comments_sampled": valid,
        "comments_with_account_id": with_ids,
        "party_positions": {party: dict(counts) for party, counts in sorted(party_counts.items())},
        "repeated_templates": len(clusters),
        "accounts_in_repeated_templates": len(set().union(*(group["accounts"] for group in repeated))),
        "templates_checked": min(len(templates), 500),
        "coordination_status": "review repeated text; automation unverified" if clusters else (
            "no repeated templates among identifiable comments" if with_ids >= 3 else "insufficient account identifiers"),
        "account_post_history_available": False,
    }