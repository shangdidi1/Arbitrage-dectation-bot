!/usr/bin/env python3
"""
Backpack Exchange Spot-Perpetual Arbitrage Monitor with Funding Rate Analysis
Monitors SOL, ETH, and BTC spot vs perpetual price differences for arbitrage opportunities
Includes funding rate history and current funding analysis for better decision making
"""

import requests
import time
import json
import csv
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import statistics
import os
from tabulate import tabulate
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backpack_arbitrage_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FundingData:
    """Data class for funding rate information"""
    current_funding_rate: float
    next_funding_time: datetime
    time_until_funding: str
    funding_history: List[Dict]  # Last 3 funding periods
    
@dataclass
class ArbitrageData:
    """Data class for arbitrage opportunity information"""
    timestamp: datetime
    base_asset: str  # SOL, ETH, BTC
    spot_symbol: str  # SOL_USDC
    perp_symbol: str  # SOL_USDC_PERP
    
    # Spot market data
    spot_best_bid: float
    spot_best_ask: float
    spot_bid_depth_5: float
    spot_ask_depth_5: float
    
    # Perp market data
    perp_best_bid: float
    perp_best_ask: float
    perp_bid_depth_5: float
    perp_ask_depth_5: float
    
    # Arbitrage opportunities
    # Buy spot, sell perp (spot_bid vs perp_ask)
    long_spot_short_perp_spread: float
    long_spot_short_perp_spread_pct: float
    long_spot_actionable: bool
    
    # Buy perp, sell spot (perp_bid vs spot_ask)  
    long_perp_short_spot_spread: float
    long_perp_short_spot_spread_pct: float
    long_perp_actionable: bool
    
    # Premium/Discount info
    perp_premium: float  # Positive if perp > spot
    perp_premium_pct: float
    expensive_leg: str  # "SPOT" or "PERP"
    
    # Funding data
    funding_data: Optional[FundingData] = None
    
class BackpackArbitrageMonitor:
    """Main class for monitoring Backpack Exchange spot-perp arbitrage"""
    
    def __init__(self):
        self.base_url = "https://api.backpack.exchange"
        self.base_assets = ["SOL", "ETH", "BTC"]
        self.spread_threshold = 0.04  # 0.04% threshold
        self.session = requests.Session()
        self.arbitrage_history = []
        self.csv_filename = "backpack_arbitrage_history.csv"
        
        # Trading pairs mapping
        self.trading_pairs = {
            "SOL": {"spot": "SOL_USDC", "perp": "SOL_USDC_PERP"},
            "ETH": {"spot": "ETH_USDC", "perp": "ETH_USDC_PERP"}, 
            "BTC": {"spot": "BTC_USDC", "perp": "BTC_USDC_PERP"}
        }
        
        # Initialize CSV file if it doesn't exist
        self._initialize_csv()
        
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_filename):
            headers = [
                'timestamp', 'base_asset', 'spot_symbol', 'perp_symbol',
                'spot_best_bid', 'spot_best_ask', 'spot_bid_depth_5', 'spot_ask_depth_5',
                'perp_best_bid', 'perp_best_ask', 'perp_bid_depth_5', 'perp_ask_depth_5',
                'long_spot_short_perp_spread', 'long_spot_short_perp_spread_pct', 'long_spot_actionable',
                'long_perp_short_spot_spread', 'long_perp_short_spot_spread_pct', 'long_perp_actionable',
                'perp_premium', 'perp_premium_pct', 'expensive_leg', 'current_funding_rate'
            ]
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            logger.info(f"Created new CSV file: {self.csv_filename}")
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Fetch orderbook data from Backpack Exchange"""
        try:
            url = f"{self.base_url}/api/v1/depth"
            params = {"symbol": symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if "bids" not in data or "asks" not in data:
                logger.error(f"Invalid orderbook format for {symbol}: {data}")
                return None
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for {symbol}: {e}")
            return None
    
    def get_funding_data(self, perp_symbol: str) -> Optional[FundingData]:
        """Fetch current and historical funding rate data"""
        try:
            # Get current funding rate and next funding time
            mark_url = f"{self.base_url}/api/v1/markPrices"
            params = {"symbol": perp_symbol}
            
            response = self.session.get(mark_url, params=params, timeout=10)
            response.raise_for_status()
            mark_data = response.json()
            
            if not mark_data or len(mark_data) == 0:
                logger.error(f"No mark price data for {perp_symbol}")
                return None
            
            mark_info = mark_data[0]
            current_funding_rate = float(mark_info.get('fundingRate', 0))
            next_funding_timestamp = int(mark_info.get('nextFundingTimestamp', 0))
            
            # Convert to datetime and calculate time until funding
            next_funding_time = datetime.fromtimestamp(next_funding_timestamp / 1000)
            time_until_funding = next_funding_time - datetime.now()
            hours_until = time_until_funding.total_seconds() / 3600
            time_until_str = f"{int(hours_until)}h {int((hours_until % 1) * 60)}m"
            
            # Get funding rate history (last 3 periods)
            history_url = f"{self.base_url}/api/v1/fundingRates"
            history_params = {"symbol": perp_symbol, "limit": 3}
            
            history_response = self.session.get(history_url, params=history_params, timeout=10)
            history_response.raise_for_status()
            funding_history = history_response.json()
            
            return FundingData(
                current_funding_rate=current_funding_rate,
                next_funding_time=next_funding_time,
                time_until_funding=time_until_str,
                funding_history=funding_history
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch funding data for {perp_symbol}: {e}")
            return None
    
    def process_orderbook(self, orderbook: Dict, symbol: str) -> Optional[Dict]:
        """Process orderbook data and return best bid/ask with depth"""
        try:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            if not bids or not asks:
                logger.warning(f"Empty orderbook for {symbol}")
                return None
            
            # Sort orderbook to ensure correct order
            bids_sorted = sorted(bids, key=lambda x: float(x[0]), reverse=True)
            asks_sorted = sorted(asks, key=lambda x: float(x[0]), reverse=False)
            
            # Get best bid and ask
            best_bid = float(bids_sorted[0][0])
            best_ask = float(asks_sorted[0][0])
            
            # Validate that bid < ask
            if best_bid >= best_ask:
                logger.warning(f"Invalid orderbook for {symbol}: bid={best_bid}, ask={best_ask}")
                return None
            
            # Calculate depth for top 5 levels
            bid_depth_5 = sum(float(bid[0]) * float(bid[1]) for bid in bids_sorted[:5])
            ask_depth_5 = sum(float(ask[0]) * float(ask[1]) for ask in asks_sorted[:5])
            
            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'bid_depth_5': bid_depth_5,
                'ask_depth_5': ask_depth_5,
                'bids': bids_sorted,
                'asks': asks_sorted
            }
            
        except (ValueError, IndexError, KeyError) as e:
            logger.error(f"Error processing orderbook for {symbol}: {e}")
            return None
    
    def calculate_arbitrage_opportunities(self, base_asset: str, spot_data: Dict, perp_data: Dict, funding_data: Optional[FundingData]) -> Optional[ArbitrageData]:
        """Calculate arbitrage opportunities between spot and perpetual markets"""
        try:
            spot_symbol = self.trading_pairs[base_asset]["spot"]
            perp_symbol = self.trading_pairs[base_asset]["perp"]
            
            # Calculate mid prices
            spot_mid = (spot_data['best_bid'] + spot_data['best_ask']) / 2
            perp_mid = (perp_data['best_bid'] + perp_data['best_ask']) / 2
            
            # Calculate perp premium/discount
            perp_premium = perp_mid - spot_mid
            perp_premium_pct = (perp_premium / spot_mid) * 100
            expensive_leg = "PERP" if perp_premium > 0 else "SPOT"
            
            # Strategy 1: Buy spot at spot_ask, sell perp at perp_bid
            # Profit = perp_bid - spot_ask
            long_spot_short_perp_spread = perp_data['best_bid'] - spot_data['best_ask']
            long_spot_short_perp_spread_pct = (long_spot_short_perp_spread / spot_data['best_ask']) * 100
            long_spot_actionable = long_spot_short_perp_spread_pct > self.spread_threshold
            
            # Strategy 2: Buy perp at perp_ask, sell spot at spot_bid  
            # Profit = spot_bid - perp_ask
            long_perp_short_spot_spread = spot_data['best_bid'] - perp_data['best_ask']
            long_perp_short_spot_spread_pct = (long_perp_short_spot_spread / perp_data['best_ask']) * 100
            long_perp_actionable = long_perp_short_spot_spread_pct > self.spread_threshold
            
            return ArbitrageData(
                timestamp=datetime.now(),
                base_asset=base_asset,
                spot_symbol=spot_symbol,
                perp_symbol=perp_symbol,
                
                spot_best_bid=spot_data['best_bid'],
                spot_best_ask=spot_data['best_ask'],
                spot_bid_depth_5=spot_data['bid_depth_5'],
                spot_ask_depth_5=spot_data['ask_depth_5'],
                
                perp_best_bid=perp_data['best_bid'],
                perp_best_ask=perp_data['best_ask'],
                perp_bid_depth_5=perp_data['bid_depth_5'],
                perp_ask_depth_5=perp_data['ask_depth_5'],
                
                long_spot_short_perp_spread=long_spot_short_perp_spread,
                long_spot_short_perp_spread_pct=long_spot_short_perp_spread_pct,
                long_spot_actionable=long_spot_actionable,
                
                long_perp_short_spot_spread=long_perp_short_spot_spread,
                long_perp_short_spot_spread_pct=long_perp_short_spot_spread_pct,
                long_perp_actionable=long_perp_actionable,
                
                perp_premium=perp_premium,
                perp_premium_pct=perp_premium_pct,
                expensive_leg=expensive_leg,
                
                funding_data=funding_data
            )
            
        except (ValueError, KeyError) as e:
            logger.error(f"Error calculating arbitrage for {base_asset}: {e}")
            return None
    
    def log_arbitrage_data(self, arb_data: ArbitrageData):
        """Log arbitrage data to CSV file"""
        try:
            row = [
                arb_data.timestamp.isoformat(),
                arb_data.base_asset,
                arb_data.spot_symbol,
                arb_data.perp_symbol,
                arb_data.spot_best_bid,
                arb_data.spot_best_ask,
                arb_data.spot_bid_depth_5,
                arb_data.spot_ask_depth_5,
                arb_data.perp_best_bid,
                arb_data.perp_best_ask,
                arb_data.perp_bid_depth_5,
                arb_data.perp_ask_depth_5,
                arb_data.long_spot_short_perp_spread,
                arb_data.long_spot_short_perp_spread_pct,
                arb_data.long_spot_actionable,
                arb_data.long_perp_short_spot_spread,
                arb_data.long_perp_short_spot_spread_pct,
                arb_data.long_perp_actionable,
                arb_data.perp_premium,
                arb_data.perp_premium_pct,
                arb_data.expensive_leg,
                arb_data.funding_data.current_funding_rate if arb_data.funding_data else 0
            ]
            
            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
        except Exception as e:
            logger.error(f"Error logging arbitrage data: {e}")
    
    def format_spread_color(self, spread_pct: float, actionable: bool) -> str:
        """Format spread percentage with color based on actionability"""
        if actionable:
            return f"{Fore.GREEN}{spread_pct:+.4f}%{Style.RESET_ALL}"
        elif spread_pct > 0:
            return f"{Fore.YELLOW}{spread_pct:+.4f}%{Style.RESET_ALL}"
        else:
            return f"{Fore.RED}{spread_pct:+.4f}%{Style.RESET_ALL}"
    
    def format_funding_rate(self, rate: float) -> str:
        """Format funding rate with color (positive = red for longs, green for shorts)"""
        rate_pct = rate * 100  # Convert to percentage
        if rate > 0:
            return f"{Fore.RED}{rate_pct:+.4f}%{Style.RESET_ALL}"
        elif rate < 0:
            return f"{Fore.GREEN}{rate_pct:+.4f}%{Style.RESET_ALL}"
        else:
            return f"{rate_pct:.4f}%"
    
    def monitor_single_cycle(self) -> List[ArbitrageData]:
        """Monitor all asset pairs for one cycle with enhanced display"""
        cycle_data = []
        
        print("\n" + "="*150)
        print(f"{Fore.CYAN}BACKPACK SPOT-PERP ARBITRAGE MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        print("="*150)
        
        # Collect all data first
        all_arb_data = []
        
        for base_asset in self.base_assets:
            spot_symbol = self.trading_pairs[base_asset]["spot"]
            perp_symbol = self.trading_pairs[base_asset]["perp"]
            
            # Fetch orderbooks
            spot_orderbook = self.get_orderbook(spot_symbol)
            perp_orderbook = self.get_orderbook(perp_symbol)
            
            if not spot_orderbook or not perp_orderbook:
                logger.warning(f"Failed to fetch orderbooks for {base_asset}")
                continue
            
            # Process orderbooks
            spot_data = self.process_orderbook(spot_orderbook, spot_symbol)
            perp_data = self.process_orderbook(perp_orderbook, perp_symbol)
            
            if not spot_data or not perp_data:
                logger.warning(f"Failed to process orderbooks for {base_asset}")
                continue
            
            # Get funding data
            funding_data = self.get_funding_data(perp_symbol)
            
            # Calculate arbitrage
            arb_data = self.calculate_arbitrage_opportunities(base_asset, spot_data, perp_data, funding_data)
            if not arb_data:
                continue
            
            # Add to history and log
            self.arbitrage_history.append(arb_data)
            self.log_arbitrage_data(arb_data)
            cycle_data.append(arb_data)
            all_arb_data.append((arb_data, spot_data, perp_data))
        
        # Display premium/discount analysis
        self._display_premium_analysis(all_arb_data)
        
        # Display funding rate analysis
        self._display_funding_analysis(all_arb_data)
        
        # Display spread and strategy summary
        self._display_strategy_summary(all_arb_data)
        
        # Display actionable opportunities with funding considerations
        self._display_actionable_opportunities(cycle_data)
        
        return cycle_data
    
    def _display_premium_analysis(self, all_arb_data: List[Tuple[ArbitrageData, Dict, Dict]]):
        """Display which market is trading at premium/discount"""
        print(f"\n{Fore.CYAN}📊 MARKET PREMIUM/DISCOUNT ANALYSIS{Style.RESET_ALL}")
        print("─" * 150)
        
        table_data = []
        headers = ["Asset", "Spot Mid", "Perp Mid", "Premium", "Premium %", "Expensive Leg", "Market Status"]
        
        for arb_data, spot_data, perp_data in all_arb_data:
            spot_mid = (arb_data.spot_best_bid + arb_data.spot_best_ask) / 2
            perp_mid = (arb_data.perp_best_bid + arb_data.perp_best_ask) / 2
            
            # Determine market status
            if abs(arb_data.perp_premium_pct) < 0.01:
                market_status = "🟢 BALANCED"
            elif arb_data.expensive_leg == "PERP":
                market_status = "📈 PERP PREMIUM (Spot Cheap)"
            else:
                market_status = "📉 SPOT PREMIUM (Perp Cheap)"
            
            # Color code the premium
            if abs(arb_data.perp_premium_pct) > 0.05:
                premium_str = f"{Fore.YELLOW}{arb_data.perp_premium:+.4f}{Style.RESET_ALL}"
                premium_pct_str = f"{Fore.YELLOW}{arb_data.perp_premium_pct:+.4f}%{Style.RESET_ALL}"
            else:
                premium_str = f"{arb_data.perp_premium:+.4f}"
                premium_pct_str = f"{arb_data.perp_premium_pct:+.4f}%"
            
            table_data.append([
                arb_data.base_asset,
                f"${spot_mid:.4f}",
                f"${perp_mid:.4f}",
                premium_str,
                premium_pct_str,
                arb_data.expensive_leg,
                market_status
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid", numalign="right"))
    
    def _display_funding_analysis(self, all_arb_data: List[Tuple[ArbitrageData, Dict, Dict]]):
        """Display funding rate analysis"""
        print(f"\n{Fore.CYAN}💰 FUNDING RATE ANALYSIS{Style.RESET_ALL}")
        print("─" * 150)
        
        table_data = []
        headers = ["Asset", "Current Rate", "Next Payment", "Last 3 Periods", "Avg Rate", "Strategy Impact"]
        
        for arb_data, spot_data, perp_data in all_arb_data:
            if not arb_data.funding_data:
                continue
            
            funding = arb_data.funding_data
            
            # Format last 3 periods
            last_3_rates = []
            for hist in funding.funding_history[:3]:
                rate = float(hist.get('fundingRate', 0)) * 100
                last_3_rates.append(f"{rate:+.4f}%")
            last_3_str = " | ".join(last_3_rates) if last_3_rates else "N/A"
            
            # Calculate average rate
            if funding.funding_history:
                avg_rate = sum(float(h.get('fundingRate', 0)) for h in funding.funding_history[:3]) / len(funding.funding_history[:3])
                avg_rate_str = self.format_funding_rate(avg_rate)
            else:
                avg_rate_str = "N/A"
            
            # Strategy impact
            current_rate = funding.current_funding_rate
            if current_rate > 0:
                impact = "📈 S1 ✓ (receive) | S2 ✗ (pay)"
            elif current_rate < 0:
                impact = "📉 S1 ✗ (pay) | S2 ✓ (receive)"
            else:
                impact = "➖ Neutral"
            
            table_data.append([
                arb_data.base_asset,
                self.format_funding_rate(current_rate),
                funding.time_until_funding,
                last_3_str,
                avg_rate_str,
                impact
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid", numalign="right"))
        
        # Add funding strategy guide
        print(f"\n{Fore.YELLOW}📌 Funding Strategy Guide:{Style.RESET_ALL}")
        print("   • S1 (Long Spot, Short Perp): Positive funding = RECEIVE | Negative funding = PAY")
        print("   • S2 (Long Perp, Short Spot): Positive funding = PAY | Negative funding = RECEIVE")
    
    def _display_strategy_summary(self, all_arb_data: List[Tuple[ArbitrageData, Dict, Dict]]):
        """Display strategy summary with clear trade directions"""
        print(f"\n{Fore.CYAN}📈 STRATEGY SUMMARY{Style.RESET_ALL}")
        print("─" * 150)
        
        table_data = []
        headers = ["Asset", "Strategy 1 (S1)", "S1 Spread", "Strategy 2 (S2)", "S2 Spread", "Recommended"]
        
        for arb_data, spot_data, perp_data in all_arb_data:
            # Strategy 1 details
            s1_details = f"Buy Spot@${arb_data.spot_best_ask:.2f} | Sell Perp@${arb_data.perp_best_bid:.2f}"
            s1_spread = self.format_spread_color(arb_data.long_spot_short_perp_spread_pct, arb_data.long_spot_actionable)
            
            # Strategy 2 details
            s2_details = f"Buy Perp@${arb_data.perp_best_ask:.2f} | Sell Spot@${arb_data.spot_best_bid:.2f}"
            s2_spread = self.format_spread_color(arb_data.long_perp_short_spot_spread_pct, arb_data.long_perp_actionable)
            
            # Recommendation based on spread and funding
            recommendation = self._get_strategy_recommendation(arb_data)
            
            table_data.append([
                arb_data.base_asset,
                s1_details,
                s1_spread,
                s2_details,
                s2_spread,
                recommendation
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid", numalign="right"))
    
    def _get_strategy_recommendation(self, arb_data: ArbitrageData) -> str:
        """Get strategy recommendation based on spread and funding"""
        if not arb_data.long_spot_actionable and not arb_data.long_perp_actionable:
            return "⚪ None"
        
        # If only one is actionable, easy choice
        if arb_data.long_spot_actionable and not arb_data.long_perp_actionable:
            return "🔵 S1"
        elif arb_data.long_perp_actionable and not arb_data.long_spot_actionable:
            return "🟣 S2"
        
        # Both actionable - consider funding
        if arb_data.funding_data:
            funding_rate = arb_data.funding_data.current_funding_rate
            
            # Positive funding favors S1 (short perp receives funding)
            # Negative funding favors S2 (long perp receives funding)
            if funding_rate > 0.0001:  # Significantly positive
                return "🔵 S1 (funding favorable)"
            elif funding_rate < -0.0001:  # Significantly negative
                return "🟣 S2 (funding favorable)"
        
        # If funding neutral or no data, choose higher spread
        if arb_data.long_spot_short_perp_spread_pct > arb_data.long_perp_short_spot_spread_pct:
            return "🔵 S1 (higher spread)"
        else:
            return "🟣 S2 (higher spread)"
    
    def _display_actionable_opportunities(self, cycle_data: List[ArbitrageData]):
        """Display actionable opportunities with funding considerations"""
        strategy1_opportunities = [data for data in cycle_data if data.long_spot_actionable]
        strategy2_opportunities = [data for data in cycle_data if data.long_perp_actionable]
        
        total_opportunities = len(strategy1_opportunities) + len(strategy2_opportunities)
        
        print(f"\n{Fore.CYAN}🎯 ACTIONABLE OPPORTUNITIES WITH FUNDING ANALYSIS{Style.RESET_ALL}")
        print("─" * 150)
        
        if total_opportunities > 0:
            if strategy1_opportunities:
                print(f"\n{Fore.GREEN}📈 Strategy 1 - Long Spot, Short Perp ({len(strategy1_opportunities)} opportunities):{Style.RESET_ALL}")
                for data in strategy1_opportunities:
                    max_trade = min(data.spot_ask_depth_5, data.perp_bid_depth_5)
                    estimated_profit = max_trade * (data.long_spot_short_perp_spread_pct / 100)
                    
                    # Funding consideration
                    funding_impact = ""
                    if data.funding_data:
                        rate = data.funding_data.current_funding_rate
                        funding_income = max_trade * rate  # Income per funding period
                        if rate > 0:
                            funding_impact = f"| {Fore.GREEN}+${funding_income:.2f}/8h funding income{Style.RESET_ALL}"
                        elif rate < 0:
                            funding_impact = f"| {Fore.RED}-${abs(funding_income):.2f}/8h funding cost{Style.RESET_ALL}"
                    
                    print(f"   • {data.base_asset}: {data.long_spot_short_perp_spread_pct:+.4f}% spread | Max: ${max_trade:,.0f} | Est: ${estimated_profit:,.2f} {funding_impact}")
                    print(f"     └─ {data.expensive_leg} trading at premium | Next funding: {data.funding_data.time_until_funding if data.funding_data else 'N/A'}")
            
            if strategy2_opportunities:
                print(f"\n{Fore.GREEN}📉 Strategy 2 - Long Perp, Short Spot ({len(strategy2_opportunities)} opportunities):{Style.RESET_ALL}")
                for data in strategy2_opportunities:
                    max_trade = min(data.perp_ask_depth_5, data.spot_bid_depth_5)
                    estimated_profit = max_trade * (data.long_perp_short_spot_spread_pct / 100)
                    
                    # Funding consideration
                    funding_impact = ""
                    if data.funding_data:
                        rate = data.funding_data.current_funding_rate
                        funding_cost = max_trade * rate  # Cost per funding period
                        if rate > 0:
                            funding_impact = f"| {Fore.RED}-${funding_cost:.2f}/8h funding cost{Style.RESET_ALL}"
                        elif rate < 0:
                            funding_impact = f"| {Fore.GREEN}+${abs(funding_cost):.2f}/8h funding income{Style.RESET_ALL}"
                    
                    print(f"   • {data.base_asset}: {data.long_perp_short_spot_spread_pct:+.4f}% spread | Max: ${max_trade:,.0f} | Est: ${estimated_profit:,.2f} {funding_impact}")
                    print(f"     └─ {data.expensive_leg} trading at premium | Next funding: {data.funding_data.time_until_funding if data.funding_data else 'N/A'}")
        else:
            print(f"{Fore.YELLOW}ℹ️  No actionable arbitrage opportunities in this cycle (threshold: {self.spread_threshold}%){Style.RESET_ALL}")
        
        print("\n" + "="*150)
    
    def generate_arbitrage_analysis(self) -> Dict:
        """Generate comprehensive analysis of arbitrage patterns"""
        if not os.path.exists(self.csv_filename):
            return {"error": "No historical data available"}
        
        try:
            df = pd.read_csv(self.csv_filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            analysis = {}
            
            for base_asset in self.base_assets:
                asset_data = df[df['base_asset'] == base_asset].copy()
                
                if len(asset_data) == 0:
                    continue
                
                analysis[base_asset] = {
                    'total_observations': len(asset_data),
                    
                    # Strategy 1 analysis (Long Spot, Short Perp)
                    'strategy1_actionable_count': asset_data['long_spot_actionable'].sum(),
                    'strategy1_actionable_percentage': (asset_data['long_spot_actionable'].sum() / len(asset_data)) * 100,
                    'strategy1_avg_spread_pct': asset_data['long_spot_short_perp_spread_pct'].mean(),
                    'strategy1_max_spread_pct': asset_data['long_spot_short_perp_spread_pct'].max(),
                    'strategy1_min_spread_pct': asset_data['long_spot_short_perp_spread_pct'].min(),
                    
                    # Strategy 2 analysis (Long Perp, Short Spot)
                    'strategy2_actionable_count': asset_data['long_perp_actionable'].sum(),
                    'strategy2_actionable_percentage': (asset_data['long_perp_actionable'].sum() / len(asset_data)) * 100,
                    'strategy2_avg_spread_pct': asset_data['long_perp_short_spot_spread_pct'].mean(),
                    'strategy2_max_spread_pct': asset_data['long_perp_short_spot_spread_pct'].max(),
                    'strategy2_min_spread_pct': asset_data['long_perp_short_spot_spread_pct'].min(),
                    
                    # Premium analysis
                    'avg_perp_premium_pct': asset_data['perp_premium_pct'].mean(),
                    'perp_premium_frequency': (asset_data['expensive_leg'] == 'PERP').sum() / len(asset_data) * 100,
                    
                    # Depth analysis
                    'avg_spot_bid_depth': asset_data['spot_bid_depth_5'].mean(),
                    'avg_spot_ask_depth': asset_data['spot_ask_depth_5'].mean(),
                    'avg_perp_bid_depth': asset_data['perp_bid_depth_5'].mean(),
                    'avg_perp_ask_depth': asset_data['perp_ask_depth_5'].mean(),
                    
                    # Funding analysis
                    'avg_funding_rate': asset_data['current_funding_rate'].mean() if 'current_funding_rate' in asset_data else 0,
                }
            
            # Overall market analysis
            total_strategy1_opportunities = sum(analysis[asset]['strategy1_actionable_count'] for asset in self.base_assets if asset in analysis)
            total_strategy2_opportunities = sum(analysis[asset]['strategy2_actionable_count'] for asset in self.base_assets if asset in analysis)
            
            analysis['market_summary'] = {
                'total_strategy1_opportunities': total_strategy1_opportunities,
                'total_strategy2_opportunities': total_strategy2_opportunities,
                'total_opportunities': total_strategy1_opportunities + total_strategy2_opportunities,
                'most_actionable_asset_strategy1': max(analysis.keys(), key=lambda x: analysis[x].get('strategy1_actionable_count', 0) if x != 'market_summary' else 0),
                'most_actionable_asset_strategy2': max(analysis.keys(), key=lambda x: analysis[x].get('strategy2_actionable_count', 0) if x != 'market_summary' else 0),
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            return {"error": str(e)}
    
    def print_analysis_report(self):
        """Print comprehensive arbitrage analysis report"""
        analysis = self.generate_arbitrage_analysis()
        
        if "error" in analysis:
            logger.error(f"Analysis error: {analysis['error']}")
            return
        
        print("\n" + "=" * 150)
        print(f"{Fore.CYAN}BACKPACK SPOT-PERP ARBITRAGE ANALYSIS REPORT{Style.RESET_ALL}")
        print("=" * 150)
        
        # Summary table
        summary_data = []
        headers = ["Asset", "Total Obs", "S1 Opps", "S1 Avg", "S2 Opps", "S2 Avg", "Avg Premium", "Avg Funding", "Avg Liquidity"]
        
        for base_asset in self.base_assets:
            if base_asset not in analysis:
                continue
                
            data = analysis[base_asset]
            avg_liquidity = (data['avg_spot_bid_depth'] + data['avg_spot_ask_depth'] + 
                           data['avg_perp_bid_depth'] + data['avg_perp_ask_depth']) / 4
            
            summary_data.append([
                base_asset,
                data['total_observations'],
                f"{data['strategy1_actionable_count']} ({data['strategy1_actionable_percentage']:.1f}%)",
                f"{data['strategy1_avg_spread_pct']:.4f}%",
                f"{data['strategy2_actionable_count']} ({data['strategy2_actionable_percentage']:.1f}%)",
                f"{data['strategy2_avg_spread_pct']:.4f}%",
                f"{data['avg_perp_premium_pct']:.4f}%",
                f"{data['avg_funding_rate']*100:.4f}%",
                f"${avg_liquidity:,.0f}"
            ])
        
        print(tabulate(summary_data, headers=headers, tablefmt="grid"))
        
        if 'market_summary' in analysis:
            summary = analysis['market_summary']
            print(f"\n{Fore.CYAN}🎯 Market Summary:{Style.RESET_ALL}")
            print(f"   Total Arbitrage Opportunities: {summary['total_opportunities']}")
            print(f"   Best Asset for Strategy 1: {summary['most_actionable_asset_strategy1']}")
            print(f"   Best Asset for Strategy 2: {summary['most_actionable_asset_strategy2']}")
        
        print("\n" + "=" * 150)
    
    def run_continuous_monitoring(self, interval_seconds: int = 1, max_cycles: int = None):
        """Run continuous arbitrage monitoring with enhanced display"""
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                
                # Monitor current cycle
                cycle_data = self.monitor_single_cycle()
                
                # Generate analysis every 10 cycles
                if cycle_count % 10 == 0:
                    self.print_analysis_report()
                
                # Check if we've reached max cycles
                if max_cycles and cycle_count >= max_cycles:
                    break
                
                # Wait for next cycle
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("\nMonitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in continuous monitoring: {e}")
        finally:
            logger.info(f"Monitoring completed after {cycle_count} cycles")
            self.print_analysis_report()

def main():
    """Main execution function"""
    print(f"{Fore.CYAN}🚀 Backpack Spot-Perp Arbitrage Monitor with Funding Analysis Starting...{Style.RESET_ALL}")
    
    monitor = BackpackArbitrageMonitor()
    
    # Run continuous monitoring
    monitor.run_continuous_monitoring(interval_seconds=1, max_cycles=100)

if __name__ == "__main__":
    main()
