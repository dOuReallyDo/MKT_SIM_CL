from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class MarketData(BaseModel):
    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

class Transaction(BaseModel):
    agent_id: str
    symbol: str
    action: str
    quantity: int
    price: float
    timestamp: datetime = Field(default_factory=datetime.now)

class AgentPerformance(BaseModel):
    agent_id: str
    initial_capital: float
    current_capital: float
    percentage_return: float
    total_trades: int
    successful_trades: int

class SimulationConfig(BaseModel):
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float
    strategy: str
    num_agents: int = 5

class SimulationResult(BaseModel):
    transactions: List[Transaction]
    agent_performances: List[AgentPerformance]
    market_data: Dict[str, List[MarketData]]
    summary: Dict[str, float] 