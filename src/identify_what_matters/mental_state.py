"""Mental State & Psychological Archetype taxonomy for Indian civic discourse.

Archetypes grounded in behavioural psychology and Indian socio-political communication:
  * CIVIC_ANGER: Institutional skepticism, anti-corruption, rage at governance failure, demands for accountability.
  * ECONOMIC_ANXIETY: Insecurity over layoffs, inflation, taxation, salary stagnation, EMI pressure.
  * PATRIOTIC_PRIDE: Cultural affinity, national prestige, sporting celebration, civilizational defense.
  * VIRAL_CYNICISM: Memetic irony, sarcastic mockery, pop-culture dismissiveness, detached amusement.
  * COMMUNITY_EMPATHY: Grassroots solidarity, mutual support, civic volunteerism, family and social care.
  * OBJECTIVE_OBSERVANT: Fact-seeking, neutral query, legal/statutory curiosity, procedural analysis.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class MentalStateArchetype(str, Enum):
    CIVIC_ANGER = "Civic Anger & Institutional Skepticism"
    ECONOMIC_ANXIETY = "Financial Anxiety & Economic Strain"
    PATRIOTIC_PRIDE = "Cultural Pride & National Solidarity"
    VIRAL_CYNICISM = "Ironic Detachment & Memetic Cynicism"
    COMMUNITY_EMPATHY = "Social Empathy & Grassroots Care"
    OBJECTIVE_OBSERVANT = "Objective Deliberation & Fact-Checking"


class MentalStateResult(NamedTuple):
    primary_state: MentalStateArchetype
    confidence: float
    secondary_state: MentalStateArchetype | None
    state_probabilities: dict[str, float]
    markers: list[str]
