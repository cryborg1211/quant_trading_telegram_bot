import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
from src.data.db_engine import DuckDBEngine

class MonthlyPostMortem:
    def __init__(self):
        # Setup directories
        self.reports_dir = "reports"
        self.images_dir = os.path.join(self.reports_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Init DB and LLM
        self.db = DuckDBEngine()
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.llm_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.llm_model = None
            print("WARNING: GEMINI_API_KEY not found. LLM Analysis will be disabled.")

    def run_analysis(self, month: int = None, year: int = None):
        """Generates the Monthly Post-Mortem Report."""
        now = datetime.now()
        month = month or now.month
        year = year or now.year
        
        print(f"📊 Starting Post-Mortem Analysis for {month:02d}/{year}...")
        
        # 1. DATA EXTRACTION
        query = f"""
            SELECT t.ticker, t.date, t.action, t.price, t.pnl_percent, m.features_snapshot
            FROM trade_history t
            LEFT JOIN rl_mistake_logs m ON t.ticker = m.ticker AND t.date = m.predicted_date
            WHERE EXTRACT(MONTH FROM CAST(t.date AS DATE)) = {month} 
              AND EXTRACT(YEAR FROM CAST(t.date AS DATE)) = {year}
              AND t.pnl_percent < 0
            ORDER BY t.pnl_percent ASC
            LIMIT 5
        """
        try:
            worst_trades = self.db.query(query)
        except Exception as e:
            print(f"❌ DB Query Failed: {e}")
            return

        if worst_trades.empty:
            print("✅ No negative trades found for this month! Amazing.")
            return

        md_content = f"# 🚨 BÁO CÁO POST-MORTEM THÁNG {month:02d}/{year}\n\n"
        md_content += f"*Phân tích tự động 5 sai lầm giao dịch lớn nhất (RL/LSTM Meta-Controller)*\n\n---\n\n"

        for idx, row in worst_trades.iterrows():
            ticker = row['ticker']
            date = row['date']
            pnl = row['pnl_percent'] * 100
            features_json_str = row['features_snapshot']
            
            print(f"Processing Trade: {ticker} (PnL: {pnl:.2f}%) on {date}")
            
            # 2. VISUALIZATION (SHAP)
            image_path = self._generate_shap_plot(ticker, date, features_json_str, idx)
            
            # 3. LLM EXCUSE GENERATOR
            explanation = self._generate_llm_explanation(ticker, date, row['action'], pnl, features_json_str)
            
            # Append to Markdown
            md_content += f"## {idx+1}. {ticker} | {row['action']} @ {row['price']:.0f} VND | Lỗ: {pnl:.2f}%\n"
            md_content += f"**Ngày Giao Dịch:** {date}\n\n"
            
            if image_path:
                # Use relative path for markdown
                rel_img_path = f"images/{os.path.basename(image_path)}"
                md_content += f"### Biểu đồ SHAP (Động lực dự báo):\n![SHAP {ticker}]({rel_img_path})\n\n"
                
            md_content += f"### Nhận định từ Quant Risk Manager (AI):\n{explanation}\n\n---\n\n"

        # 4. REPORT EXPORT
        report_path = os.path.join(self.reports_dir, f"postmortem_{month:02d}_{year}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        print(f"✅ Report successfully generated at: {report_path}")

    def _generate_shap_plot(self, ticker, date, features_json_str, idx):
        """Reconstructs features and saves a SHAP waterfall plot."""
        if not features_json_str or pd.isna(features_json_str):
            return None
            
        try:
            feature_data = json.loads(features_json_str)
            # Assuming features_snapshot stores a dictionary of {feature_name: shap_value}
            # Since this is a post-mortem analytical tool, if data isn't perfect, we mock the visualization 
            # structure using exact keys from the snapshot.
            
            if not isinstance(feature_data, dict):
                feature_data = {"Unknown_Feature": 0.0}

            features = list(feature_data.keys())
            shap_values = np.array(list(feature_data.values()), dtype=float)
            
            # Mock base value for the explanation object
            base_value = 0.5 
            
            explanation = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=np.zeros_like(shap_values), # Dummy raw data
                feature_names=features
            )
            
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(explanation, show=False)
            
            clean_date = str(date).replace("-", "")
            img_filename = f"shap_{ticker}_{clean_date}_{idx}.png"
            img_path = os.path.join(self.images_dir, img_filename)
            
            plt.savefig(img_path, bbox_inches='tight', dpi=150)
            plt.close()
            return img_path
            
        except Exception as e:
            print(f"  [SHAP] Error generating plot for {ticker}: {e}")
            return None

    def _generate_llm_explanation(self, ticker, date, action, pnl, features_json_str):
        """Sends data to Gemini to generate a risk management 'excuse'."""
        if not self.llm_model:
            return "❌ API Key không tồn tại. Lỗi không thể gọi LLM."
            
        prompt = f"""
        You are a Quant Risk Manager. Looking at this historical data snapshot where our LSTM & RL Meta-Controller made a wrong prediction, explain in Vietnamese WHY it likely failed.
        
        - Ticker: {ticker}
        - Date: {date}
        - Action Taken: {action}
        - Actual PnL: {pnl:.2f}%
        - Features State at execution: {features_json_str}
        
        Was it a fake breakout? Did macro news override technicals? Was there a liquidity trap?
        
        MANDATORY FORMAT:
        Provide exactly a 3-bullet-point post-mortem STRICTLY IN VIETNAMESE (Tiếng Việt).
        """
        
        try:
            response = self.llm_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"❌ Lỗi truy vấn Gemini API: {e}"

if __name__ == "__main__":
    analyst = MonthlyPostMortem()
    analyst.run_analysis()
