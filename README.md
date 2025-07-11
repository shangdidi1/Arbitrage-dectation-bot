# Arbitrage-dectation-bot
#detect the perp and spot price difference of the backpack exchange, use api and update every 30sec, 
#!/usr/bin/env python3
"""
Backpack Exchange Spot-Perpetual Arbitrage Monitor
Monitors SOL, ETH, and BTC spot vs perpetual price differences for arbitrage opportunities
Calculates spread between spot bid/perp ask and perp bid/spot ask
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
    
class BackpackArbitrageMonitor:
    """Main class for monitoring Backpack Exchange spot-perp arbitrage"""
    
    def __init__(self):
        self.base_url = "https://api.backpack.exchange"
        self.base_assets = ["SOL", "ETH", "BTC"]
        self.spread_threshold = 0.06  # 0.06% threshold
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
                'long_perp_short_spot_spread', 'long_perp_short_spot_spread_pct', 'long_perp_actionable'
            ]
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            logger.info(f"Created new CSV file: {self.csv_filename}")
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """
        Fetch orderbook data from Backpack Exchange
        
        Args:
            symbol: Trading symbol (e.g., "SOL_USDC", "SOL_USDC_PERP")
            
        Returns:
            Orderbook data or None if error
        """
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
    
    def process_orderbook(self, orderbook: Dict, symbol: str) -> Optional[Dict]:
        """
        Process orderbook data and return best bid/ask with depth
        
        Args:
            orderbook: Raw orderbook data
            symbol: Trading symbol for logging
            
        Returns:
            Processed orderbook data or None if error
        """
        try:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            if not bids or not asks:
                logger.warning(f"Empty orderbook for {symbol}")
                return None
            
            # Sort orderbook to ensure correct order
            # Bids should be sorted by price descending (highest price first)
            # Asks should be sorted by price ascending (lowest price first)
            bids_sorted = sorted(bids, key=lambda x: float(x[0]), reverse=True)
            asks_sorted = sorted(asks, key=lambda x: float(x[0]), reverse=False)
            
            # Get best bid (highest price) and ask (lowest price)
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
    
    def calculate_arbitrage_opportunities(self, base_asset: str, spot_data: Dict, perp_data: Dict) -> Optional[ArbitrageData]:
        """
        Calculate arbitrage opportunities between spot and perpetual markets
        
        Args:
            base_asset: Base asset (SOL, ETH, BTC)
            spot_data: Processed spot orderbook data
            perp_data: Processed perp orderbook data
            
        Returns:
            ArbitrageData object or None if calculation fails
        """
        try:
            spot_symbol = self.trading_pairs[base_asset]["spot"]
            perp_symbol = self.trading_pairs[base_asset]["perp"]
            
            # Strategy 1: Buy spot, sell perp (long spot, short perp)
            # Compare spot ask price (what we pay to buy spot) vs perp bid price (what we receive selling perp)
            # Actually, for arbitrage we want: spot bid (what we can sell spot for) vs perp ask (what we pay for perp)
            # Wait, let me think about this correctly:
            
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
                long_perp_actionable=long_perp_actionable
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
                arb_data.long_perp_actionable
            ]
            
            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
        except Exception as e:
            logger.error(f"Error logging arbitrage data: {e}")
    
    def monitor_single_cycle(self) -> List[ArbitrageData]:
        """
        Monitor all asset pairs for one cycle
        
        Returns:
            List of ArbitrageData objects
        """
        cycle_data = []
        
        logger.info("=" * 80)
        logger.info("BACKPACK SPOT-PERP ARBITRAGE MONITOR - CYCLE START")
        logger.info("=" * 80)
        
        for base_asset in self.base_assets:
            logger.info(f"Analyzing {base_asset} arbitrage opportunities...")
            
            spot_symbol = self.trading_pairs[base_asset]["spot"]
            perp_symbol = self.trading_pairs[base_asset]["perp"]
            
            # Fetch orderbooks for both markets
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
            
            # Calculate arbitrage opportunities
            arb_data = self.calculate_arbitrage_opportunities(base_asset, spot_data, perp_data)
            if not arb_data:
                continue
            
            # Add to history and log to CSV
            self.arbitrage_history.append(arb_data)
            self.log_arbitrage_data(arb_data)
            cycle_data.append(arb_data)
            
            # Print current status
            self._print_arbitrage_status(arb_data)
        
        logger.info("=" * 80)
        return cycle_data
    
    def _print_arbitrage_status(self, arb_data: ArbitrageData):
        """Print detailed arbitrage status for a single asset"""
        
        # Strategy 1 status
        strategy1_status = "🟢 ACTIONABLE" if arb_data.long_spot_actionable else "🔴 NOT ACTIONABLE"
        strategy2_status = "🟢 ACTIONABLE" if arb_data.long_perp_actionable else "🔴 NOT ACTIONABLE"
        
        logger.info(f"""
📊 {arb_data.base_asset} ARBITRAGE ANALYSIS:

📈 STRATEGY 1 - Long Spot, Short Perp: {strategy1_status}
├─ Buy {arb_data.spot_symbol} at: ${arb_data.spot_best_ask:.4f}
├─ Sell {arb_data.perp_symbol} at: ${arb_data.perp_best_bid:.4f}  
├─ Spread: ${arb_data.long_spot_short_perp_spread:.4f} ({arb_data.long_spot_short_perp_spread_pct:.4f}%)
├─ Spot Ask Depth: ${arb_data.spot_ask_depth_5:,.2f}
└─ Perp Bid Depth: ${arb_data.perp_bid_depth_5:,.2f}

📉 STRATEGY 2 - Long Perp, Short Spot: {strategy2_status}
├─ Buy {arb_data.perp_symbol} at: ${arb_data.perp_best_ask:.4f}
├─ Sell {arb_data.spot_symbol} at: ${arb_data.spot_best_bid:.4f}
├─ Spread: ${arb_data.long_perp_short_spot_spread:.4f} ({arb_data.long_perp_short_spot_spread_pct:.4f}%)
├─ Perp Ask Depth: ${arb_data.perp_ask_depth_5:,.2f}
└─ Spot Bid Depth: ${arb_data.spot_bid_depth_5:,.2f}
        """)
    
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
                    
                    # Depth analysis
                    'avg_spot_bid_depth': asset_data['spot_bid_depth_5'].mean(),
                    'avg_spot_ask_depth': asset_data['spot_ask_depth_5'].mean(),
                    'avg_perp_bid_depth': asset_data['perp_bid_depth_5'].mean(),
                    'avg_perp_ask_depth': asset_data['perp_ask_depth_5'].mean(),
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
        
        print("\n" + "=" * 100)
        print("BACKPACK SPOT-PERP ARBITRAGE ANALYSIS REPORT")
        print("=" * 100)
        
        for base_asset in self.base_assets:
            if base_asset not in analysis:
                continue
                
            data = analysis[base_asset]
            print(f"\n💰 {base_asset} Arbitrage Analysis:")
            print(f"   Total Observations: {data['total_observations']}")
            
            print(f"\n   📈 Strategy 1 (Long Spot, Short Perp):")
            print(f"      Actionable Opportunities: {data['strategy1_actionable_count']} ({data['strategy1_actionable_percentage']:.2f}%)")
            print(f"      Average Spread: {data['strategy1_avg_spread_pct']:.4f}%")
            print(f"      Spread Range: {data['strategy1_min_spread_pct']:.4f}% - {data['strategy1_max_spread_pct']:.4f}%")
            
            print(f"\n   📉 Strategy 2 (Long Perp, Short Spot):")
            print(f"      Actionable Opportunities: {data['strategy2_actionable_count']} ({data['strategy2_actionable_percentage']:.2f}%)")
            print(f"      Average Spread: {data['strategy2_avg_spread_pct']:.4f}%")
            print(f"      Spread Range: {data['strategy2_min_spread_pct']:.4f}% - {data['strategy2_max_spread_pct']:.4f}%")
            
            print(f"\n   💧 Liquidity Analysis:")
            print(f"      Avg Spot Bid Depth: ${data['avg_spot_bid_depth']:,.2f}")
            print(f"      Avg Spot Ask Depth: ${data['avg_spot_ask_depth']:,.2f}")
            print(f"      Avg Perp Bid Depth: ${data['avg_perp_bid_depth']:,.2f}")
            print(f"      Avg Perp Ask Depth: ${data['avg_perp_ask_depth']:,.2f}")
        
        if 'market_summary' in analysis:
            summary = analysis['market_summary']
            print(f"\n🎯 Market Summary:")
            print(f"   Total Strategy 1 Opportunities: {summary['total_strategy1_opportunities']}")
            print(f"   Total Strategy 2 Opportunities: {summary['total_strategy2_opportunities']}")
            print(f"   Total Arbitrage Opportunities: {summary['total_opportunities']}")
            print(f"   Best Asset for Strategy 1: {summary['most_actionable_asset_strategy1']}")
            print(f"   Best Asset for Strategy 2: {summary['most_actionable_asset_strategy2']}")
        
        print("\n" + "=" * 100)
    
    def create_visualization(self):
        """Create visualization plots of arbitrage data"""
        if not os.path.exists(self.csv_filename):
            logger.warning("No data available for visualization")
            return
        
        try:
            df = pd.read_csv(self.csv_filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Backpack Spot-Perp Arbitrage Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Strategy 1 spreads over time
            ax1 = axes[0, 0]
            for asset in self.base_assets:
                asset_data = df[df['base_asset'] == asset]
                ax1.plot(asset_data['timestamp'], asset_data['long_spot_short_perp_spread_pct'], 
                        label=f'{asset} Long Spot/Short Perp', marker='o', markersize=2)
            ax1.axhline(y=self.spread_threshold, color='red', linestyle='--', alpha=0.7, label='Threshold')
            ax1.set_title('Strategy 1: Long Spot, Short Perp Spreads')
            ax1.set_ylabel('Spread (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Strategy 2 spreads over time
            ax2 = axes[0, 1]
            for asset in self.base_assets:
                asset_data = df[df['base_asset'] == asset]
                ax2.plot(asset_data['timestamp'], asset_data['long_perp_short_spot_spread_pct'], 
                        label=f'{asset} Long Perp/Short Spot', marker='o', markersize=2)
            ax2.axhline(y=self.spread_threshold, color='red', linestyle='--', alpha=0.7, label='Threshold')
            ax2.set_title('Strategy 2: Long Perp, Short Spot Spreads')
            ax2.set_ylabel('Spread (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Actionable opportunities by asset and strategy
            ax3 = axes[0, 2]
            strategy1_counts = []
            strategy2_counts = []
            for asset in self.base_assets:
                asset_data = df[df['base_asset'] == asset]
                strategy1_counts.append(asset_data['long_spot_actionable'].sum())
                strategy2_counts.append(asset_data['long_perp_actionable'].sum())
            
            x = range(len(self.base_assets))
            width = 0.35
            ax3.bar([i - width/2 for i in x], strategy1_counts, width, label='Strategy 1', alpha=0.8)
            ax3.bar([i + width/2 for i in x], strategy2_counts, width, label='Strategy 2', alpha=0.8)
            ax3.set_title('Actionable Opportunities by Strategy')
            ax3.set_ylabel('Count')
            ax3.set_xticks(x)
            ax3.set_xticklabels(self.base_assets)
            ax3.legend()
            
            # Plot 4: Spot vs Perp depth comparison
            ax4 = axes[1, 0]
            for asset in self.base_assets:
                asset_data = df[df['base_asset'] == asset]
                avg_spot_depth = (asset_data['spot_bid_depth_5'] + asset_data['spot_ask_depth_5']).mean() / 2
                avg_perp_depth = (asset_data['perp_bid_depth_5'] + asset_data['perp_ask_depth_5']).mean() / 2
                ax4.bar([f'{asset}\nSpot', f'{asset}\nPerp'], [avg_spot_depth, avg_perp_depth])
            ax4.set_title('Average Market Depth Comparison')
            ax4.set_ylabel('Depth (USD)')
            
            # Plot 5: Spread distribution for Strategy 1
            ax5 = axes[1, 1]
            for asset in self.base_assets:
                asset_data = df[df['base_asset'] == asset]
                ax5.hist(asset_data['long_spot_short_perp_spread_pct'], alpha=0.6, label=f'{asset} Strategy 1', bins=20)
            ax5.axvline(x=self.spread_threshold, color='red', linestyle='--', alpha=0.7, label='Threshold')
            ax5.set_title('Strategy 1 Spread Distribution')
            ax5.set_xlabel('Spread (%)')
            ax5.set_ylabel('Frequency')
            ax5.legend()
            
            # Plot 6: Spread distribution for Strategy 2
            ax6 = axes[1, 2]
            for asset in self.base_assets:
                asset_data = df[df['base_asset'] == asset]
                ax6.hist(asset_data['long_perp_short_spot_spread_pct'], alpha=0.6, label=f'{asset} Strategy 2', bins=20)
            ax6.axvline(x=self.spread_threshold, color='red', linestyle='--', alpha=0.7, label='Threshold')
            ax6.set_title('Strategy 2 Spread Distribution')
            ax6.set_xlabel('Spread (%)')
            ax6.set_ylabel('Frequency')
            ax6.legend()
            
            plt.tight_layout()
            plt.savefig('backpack_arbitrage_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info("Visualization saved as 'backpack_arbitrage_analysis.png'")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    
    def run_continuous_monitoring(self, interval_seconds: int = 30, max_cycles: int = None):
        """
        Run continuous arbitrage monitoring
        
        Args:
            interval_seconds: Time between monitoring cycles
            max_cycles: Maximum number of cycles (None for infinite)
        """
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                logger.info(f"Starting arbitrage monitoring cycle #{cycle_count}")
                
                # Monitor current cycle
                cycle_data = self.monitor_single_cycle()
                
                # Print actionable opportunities summary
                strategy1_opportunities = [data for data in cycle_data if data.long_spot_actionable]
                strategy2_opportunities = [data for data in cycle_data if data.long_perp_actionable]
                
                total_opportunities = len(strategy1_opportunities) + len(strategy2_opportunities)
                
                if total_opportunities > 0:
                    logger.info(f"🚨 ARBITRAGE OPPORTUNITIES FOUND: {total_opportunities}")
                    if strategy1_opportunities:
                        logger.info(f"   📈 Strategy 1 (Long Spot/Short Perp): {len(strategy1_opportunities)}")
                        for data in strategy1_opportunities:
                            logger.info(f"      {data.base_asset}: {data.long_spot_short_perp_spread_pct:.4f}% spread")
                    if strategy2_opportunities:
                        logger.info(f"   📉 Strategy 2 (Long Perp/Short Spot): {len(strategy2_opportunities)}")
                        for data in strategy2_opportunities:
                            logger.info(f"      {data.base_asset}: {data.long_perp_short_spot_spread_pct:.4f}% spread")
                else:
                    logger.info("ℹ️  No actionable arbitrage opportunities this cycle")
                
                # Generate analysis every 10 cycles
                if cycle_count % 10 == 0:
                    self.print_analysis_report()
                
                # Check if we've reached max cycles
                if max_cycles and cycle_count >= max_cycles:
                    break
                
                # Wait for next cycle
                logger.info(f"Waiting {interval_seconds} seconds until next cycle...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in continuous monitoring: {e}")
        finally:
            logger.info(f"Monitoring completed after {cycle_count} cycles")
            self.print_analysis_report()

def main():
    """Main execution function"""
    print("🚀 Backpack Spot-Perp Arbitrage Monitor Starting...")
    
    monitor = BackpackArbitrageMonitor()
    
    # You can choose different modes:
    
    # 1. Single monitoring cycle
    # monitor.monitor_single_cycle()
    # monitor.print_analysis_report()
    
    # 2. Continuous monitoring (uncomment to use)
    monitor.run_continuous_monitoring(interval_seconds=30, max_cycles=20)
    
    # 3. Generate visualization (uncomment to use)
    # monitor.create_visualization()

if __name__ == "__main__":
    main()
