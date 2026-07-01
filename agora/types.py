"""Core value objects for the Measurement Market.

Everything here is a plain, serializable dataclass. The game engine, the
market, the reward rule and the analysis code all speak in terms of these
types so that a full game can be replayed from its transcript.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(str, Enum):
    MEASURE = "measure"
    SEND_MESSAGE = "send_message"
    TRANSFER = "transfer_credits"
    PROPOSE_TRADE = "propose_trade"
    RESPOND_TRADE = "respond_trade"
    SUBMIT_ESTIMATE = "submit_estimate"
    END_TURN = "end_turn"


@dataclass
class Action:
    """A single move emitted by an agent policy during its turn."""

    type: ActionType
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "args": self.args}


@dataclass
class Message:
    sender: str
    recipient: str  # an agent id, or "all" for a broadcast
    text: str
    tick: int


@dataclass
class Trade:
    """A proposed sale of a (claimed) measurement.

    ``claimed_value`` is whatever the seller *says* it measured — it may be a
    lie. The referee separately records the value the seller actually observed
    (if any) so deception is verifiable after the fact.
    """

    trade_id: str
    seller: str
    buyer: str
    price: float
    claimed_value: float
    tick: int
    status: str = "pending"  # pending | accepted | rejected | expired


@dataclass
class Measurement:
    """One draw from the measurement tool, kept in the agent's private log."""

    agent_id: str
    value: float          # the noisy sample the agent actually saw
    truth: float          # ground truth for the round (referee-only bookkeeping)
    tau: float            # the noise std that generated this sample
    tick: int
    cost: float


@dataclass
class AgentState:
    agent_id: str
    credits: float
    tau: float                         # private measurement-noise std
    messages_left: int
    alive: bool = True
    estimate: Optional[float] = None
    measurements: List[Measurement] = field(default_factory=list)
    inbox: List[Message] = field(default_factory=list)
    # measurements bought from others: (seller, claimed_value, price)
    purchased: List[Dict[str, Any]] = field(default_factory=list)

    def can_afford(self, amount: float) -> bool:
        return self.credits >= amount - 1e-9


@dataclass
class RoundResult:
    round_index: int
    truth: float
    estimates: Dict[str, Optional[float]]
    errors: Dict[str, float]
    rewards: Dict[str, float]
    credits_start: Dict[str, float]
    credits_end: Dict[str, float]
    alive: Dict[str, bool]


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of dataclasses/enums to plain JSON types."""
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
