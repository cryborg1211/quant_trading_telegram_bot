import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.data.db_engine import DuckDBEngine

class QuantTradingEnv(gym.Env):
    """
    Reinforcement Learning Environment for the Alpha360 Meta-Controller.
    This environment sits on top of LSTM and Gemini outputs to optimize 
    dynamic position sizing and risk management.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, telegram_id="default_user", dataset=None):
        """
        Initialize the RL Environment.
        
        Args:
            telegram_id (str): User identifier for DB queries.
            dataset (list of dicts): Pre-fetched chronological state data containing 
                                     LSTM probs, sentiment, and market returns.
        """
        super(QuantTradingEnv, self).__init__()
        
        self.telegram_id = telegram_id
        self.db = DuckDBEngine()
        self.dataset = dataset if dataset is not None else []
        self.current_step = 0
        
        # State: Current Position Ratio (-1.0 to 1.0, assuming Long/Short capability, or 0 to 1 for Long only)
        # For simplicity, we assume 0.0 is Flat, 1.0 is Full Long.
        self.current_position = 0.0 

        # -------------------------------------------------------------
        # OBSERVATION SPACE (State)
        # [LSTM_Up_Probability, Gemini_Sentiment_Score, Current_Position_Ratio, Floating_PnL]
        # -------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 10.0], dtype=np.float32), # Cap floating PnL arbitrarily at 1000%
            dtype=np.float32
        )

        # -------------------------------------------------------------
        # ACTION SPACE
        # Continuous space: [-1.0, 1.0] representing target position size
        # -1.0 = Full Sell (or Short), 0 = Hold Flat, 1.0 = Full Buy
        # -------------------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Resets the environment to the beginning of the dataset."""
        super().reset(seed=seed)
        self.current_step = 0
        self.current_position = 0.0
        
        return self._get_observation(), {}

    def _get_observation(self):
        """Constructs the current state array."""
        if self.current_step >= len(self.dataset):
            return np.zeros(4, dtype=np.float32)
            
        row = self.dataset[self.current_step]
        
        lstm_prob = float(row.get('lstm_up_prob', 0.5))
        sentiment = float(row.get('sentiment_score', 0.0))
        floating_pnl = float(row.get('floating_pnl', 0.0))
        
        return np.array([
            lstm_prob, 
            sentiment, 
            self.current_position, 
            floating_pnl
        ], dtype=np.float32)

    def step(self, action):
        """
        Executes an action, calculates rewards based on DuckDB history, and advances state.
        """
        if self.current_step >= len(self.dataset):
            return self._get_observation(), 0.0, True, False, {}

        current_row = self.dataset[self.current_step]
        ticker = current_row.get('ticker')
        date_str = current_row.get('date')
        
        # 1. Action Processing
        target_position = np.clip(action[0], -1.0, 1.0)
        position_change = target_position - self.current_position
        self.current_position = target_position 
        
        # 2. Base Reward (Realized/Unrealized PnL of the Step)
        reward = 0.0
        actual_outcome = self._fetch_actual_outcome(ticker, date_str, current_row)
        
        # Step PnL is directional: if we are Long (1.0) and outcome is +5%, reward is +0.05
        # If we are Short (-1.0) and outcome is -5%, reward is +0.05
        step_pnl = self.current_position * actual_outcome
        reward += step_pnl

        # 3. Transaction Fee Penalty (0.4%)
        # Only penalize if there was a meaningful change in position
        if abs(position_change) > 0.05: 
            reward -= 0.004

        # 4. Mistake Penalty (Heavy negative reward for hitting Stop Loss)
        if actual_outcome <= -0.07:
            if self.current_position > 0.5: # We were heavily Long during a crash
                reward -= 1.0 # Significant punishment

        self.current_step += 1
        done = self.current_step >= len(self.dataset)
        
        info = {
            "step_pnl": step_pnl,
            "actual_outcome": actual_outcome,
            "position": self.current_position
        }
        
        return self._get_observation(), float(reward), done, False, info

    def _fetch_actual_outcome(self, ticker, date_str, current_row):
        """
        Queries DuckDB to find what ACTUALLY happened to the asset.
        Checks `rl_mistake_logs` first, then `trade_history`.
        """
        if not ticker or not date_str:
            return current_row.get('next_period_return', 0.0)
            
        try:
            # Check Mistake Logs (False Positives)
            mistake_query = f"""
                SELECT actual_t5_outcome FROM rl_mistake_logs 
                WHERE ticker = '{ticker}' AND predicted_date = '{date_str}'
            """
            mistake_df = self.db.query(mistake_query)
            
            if not mistake_df.empty:
                return float(mistake_df.iloc[0]['actual_t5_outcome'])

            # Check Trade History for realized PnL
            trade_query = f"""
                SELECT pnl_percent FROM trade_history 
                WHERE ticker = '{ticker}' AND date >= '{date_str}' LIMIT 1
            """
            trade_df = self.db.query(trade_query)
            
            if not trade_df.empty:
                return float(trade_df.iloc[0]['pnl_percent'])
                
        except Exception as e:
            print(f"[RL Env] DB Query Error: {e}")
            
        # Fallback to dataset if DB lacks future records
        return float(current_row.get('next_period_return', 0.0))
