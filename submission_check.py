#!/usr/bin/env python3
"""
Strategy Validation Script for EvaluatorNode
===========================================

This script performs comprehensive pre-checks on strategy.py files to ensure they meet
all platform requirements before deployment. It validates module loading, function
signatures, data formats, market availability, volume requirements, and signal generation.

Usage: python validate_strategy.py (looks for strategy.py in same directory)
"""

import sys
import os
import importlib.util
import requests
import json
import pandas as pd
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime
from data_download_manager import CryptoDataManager

# ANSI color codes for pretty output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

class StrategyValidator:
    """
    Comprehensive validator for strategy.py files.
    
    This class performs a series of validation tests to ensure the strategy
    meets all technical and business requirements of the evaluation platform.
    """
    
    def __init__(self, strategy_path: str):
        self.strategy_path = Path(strategy_path)
        self.strategy_module = None
        self.allowed_timeframes = {'1H', '2H', '4H', '12H', '1D'}
        self.max_targets = 3
        self.max_anchors = 5
        self.min_target_volume_usd = 5_000_000  # $5M
        self.min_anchor_volume_usd = 50_000_000  # $50M
        
        # Fixed date range for volume analysis and data generation
        self.volume_start_date = "2025-01-01"
        self.volume_end_date = "2025-06-30"
        
        # Expected number of 1H rows for the full date range
        self.expected_1h_rows = 4368  # 182 days × 24 hours (Jan 1, 2025 to June 30, 2025)
        
    def print_header(self):
        """Print a nice header for the validation process."""
        print("\n" + "="*60)
        print(f"{Colors.BOLD}{Colors.CYAN}Strategy Validation Suite{Colors.END}")
        print(f"Strategy File: {Colors.YELLOW}{self.strategy_path}{Colors.END}")
        print("="*60)
        
    def print_test_result(self, test_name: str, success: bool, details: str = ""):
        """Print formatted test results with colors and symbols."""
        symbol = f"{Colors.GREEN}✓{Colors.END}" if success else f"{Colors.RED}✗{Colors.END}"
        status = f"{Colors.GREEN}PASS{Colors.END}" if success else f"{Colors.RED}FAIL{Colors.END}"
        
        print(f"\n{symbol} {Colors.BOLD}{test_name}{Colors.END} - {status}")
        if details:
            # Indent details for better readability
            for line in details.split('\n'):
                if line.strip():
                    print(f"   {line}")
    
    def test_module_loading(self) -> bool:
        """
        Test 1: Attempt to load the strategy.py file as a Python module.
        
        This test verifies that the strategy file has valid Python syntax
        and can be imported without errors.
        """
        try:
            # Load the module from file path
            spec = importlib.util.spec_from_file_location("strategy", self.strategy_path)
            if spec is None or spec.loader is None:
                self.print_test_result(
                    "Module Loading", 
                    False, 
                    f"Could not create module spec for {self.strategy_path}"
                )
                return False
                
            self.strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.strategy_module)
            
            self.print_test_result(
                "Module Loading", 
                True, 
                f"Successfully loaded strategy module from {self.strategy_path.name}"
            )
            return True
            
        except Exception as e:
            self.print_test_result(
                "Module Loading", 
                False, 
                f"Failed to load module: {str(e)}"
            )
            return False
    
    def test_function_exists(self) -> bool:
        """
        Test 2: Check if get_coin_metadata() function exists in the module.
        
        This test verifies that the required function is present and callable.
        """
        try:
            if not hasattr(self.strategy_module, 'get_coin_metadata'):
                self.print_test_result(
                    "Function Existence", 
                    False, 
                    "get_coin_metadata() function not found in strategy module"
                )
                return False
            
            func = getattr(self.strategy_module, 'get_coin_metadata')
            if not callable(func):
                self.print_test_result(
                    "Function Existence", 
                    False, 
                    "get_coin_metadata exists but is not callable"
                )
                return False
            
            self.print_test_result(
                "Function Existence", 
                True, 
                "get_coin_metadata() function found and is callable"
            )
            return True
            
        except Exception as e:
            self.print_test_result(
                "Function Existence", 
                False, 
                f"Error checking function: {str(e)}"
            )
            return False
    
    def test_function_output_format(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Test 3: Validate the output format of get_coin_metadata().
        
        This test calls the function and verifies it returns data in the expected
        format with targets and anchors arrays containing symbol/timeframe objects.
        """
        try:
            # Call the function to get metadata
            metadata = self.strategy_module.get_coin_metadata()
            
            # Check if return value is a dictionary
            if not isinstance(metadata, dict):
                self.print_test_result(
                    "Output Format", 
                    False, 
                    f"Expected dict, got {type(metadata).__name__}"
                )
                return False, {}
            
            # Check for required keys
            required_keys = ['targets', 'anchors']
            missing_keys = [key for key in required_keys if key not in metadata]
            if missing_keys:
                self.print_test_result(
                    "Output Format", 
                    False, 
                    f"Missing required keys: {missing_keys}"
                )
                return False, {}
            
            # Validate targets structure
            targets = metadata['targets']
            if not isinstance(targets, list):
                self.print_test_result(
                    "Output Format", 
                    False, 
                    f"'targets' should be a list, got {type(targets).__name__}"
                )
                return False, {}
            
            # Validate anchors structure
            anchors = metadata['anchors']
            if not isinstance(anchors, list):
                self.print_test_result(
                    "Output Format", 
                    False, 
                    f"'anchors' should be a list, got {type(anchors).__name__}"
                )
                return False, {}
            
            # Validate individual target/anchor objects
            for i, target in enumerate(targets):
                if not isinstance(target, dict) or 'symbol' not in target or 'timeframe' not in target:
                    self.print_test_result(
                        "Output Format", 
                        False, 
                        f"Target {i} missing 'symbol' or 'timeframe' fields"
                    )
                    return False, {}
            
            for i, anchor in enumerate(anchors):
                if not isinstance(anchor, dict) or 'symbol' not in anchor or 'timeframe' not in anchor:
                    self.print_test_result(
                        "Output Format", 
                        False, 
                        f"Anchor {i} missing 'symbol' or 'timeframe' fields"
                    )
                    return False, {}
            
            self.print_test_result(
                "Output Format", 
                True, 
                f"Valid format with {len(targets)} targets and {len(anchors)} anchors"
            )
            return True, metadata
            
        except Exception as e:
            self.print_test_result(
                "Output Format", 
                False, 
                f"Error calling get_coin_metadata(): {str(e)}"
            )
            return False, {}
    
    def test_limits_compliance(self, metadata: Dict[str, Any]) -> bool:
        """
        Test 4: Check if the strategy complies with target/anchor limits.
        
        This test verifies that the number of targets and anchors doesn't exceed
        the platform's maximum limits.
        """
        targets = metadata.get('targets', [])
        anchors = metadata.get('anchors', [])
        
        issues = []
        
        # Check target count
        if len(targets) > self.max_targets:
            issues.append(f"Too many targets: {len(targets)} (max: {self.max_targets})")
        
        # Check anchor count
        if len(anchors) > self.max_anchors:
            issues.append(f"Too many anchors: {len(anchors)} (max: {self.max_anchors})")
        
        # Check timeframes
        all_coins = targets + anchors
        invalid_timeframes = []
        for coin in all_coins:
            if coin['timeframe'] not in self.allowed_timeframes:
                invalid_timeframes.append(f"{coin['symbol']}:{coin['timeframe']}")
        
        if invalid_timeframes:
            issues.append(f"Invalid timeframes: {', '.join(invalid_timeframes)}")
            issues.append(f"Allowed: {', '.join(sorted(self.allowed_timeframes))}")
        
        success = len(issues) == 0
        details = '\n'.join(issues) if issues else "All limits and timeframes are valid"
        
        self.print_test_result("Limits Compliance", success, details)
        return success
    
    def fetch_historical_volume_data(self, symbol: str) -> float:
        """
        Fetch historical volume data for a USDT pair using the fixed date range.
        Returns average daily volume in USD.
        """
        usdt_pair = f"{symbol}USDT"
        
        try:
            # Convert dates to timestamps
            start_ts = int(datetime.strptime(self.volume_start_date, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.strptime(self.volume_end_date, "%Y-%m-%d").timestamp() * 1000)
            
            print(f"   Fetching volume data for {usdt_pair}...")
            
            response = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    'symbol': usdt_pair,
                    'interval': '1d',  # Daily data
                    'startTime': start_ts,
                    'endTime': end_ts,
                    'limit': 1000  # More than enough for 1 year
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"   API error {response.status_code} for {usdt_pair}")
                return 0.0
            
            kline_data = response.json()
            
            if not kline_data:
                print(f"   No data returned for {usdt_pair}")
                return 0.0
            
            # Extract daily volumes (index 7 is quote asset volume in USDT)
            daily_volumes = []
            for kline in kline_data:
                try:
                    volume_usd = float(kline[7])  # USDT volume = USD volume
                    daily_volumes.append(volume_usd)
                except (ValueError, IndexError):
                    continue
            
            if daily_volumes:
                avg_volume = sum(daily_volumes) / len(daily_volumes)
                print(f"   Got {len(daily_volumes)} days of data, avg volume: ${avg_volume:,.0f}")
                return avg_volume
            else:
                print(f"   No valid volume data for {usdt_pair}")
                return 0.0
                
        except Exception as e:
            print(f"   Error fetching data for {usdt_pair}: {str(e)}")
            return 0.0
    
    def test_symbol_availability(self, metadata: Dict[str, Any]) -> bool:
        """
        Test 5: Check if all symbols exist as USDT pairs on Binance.
        """
        all_coins = metadata.get('targets', []) + metadata.get('anchors', [])
        missing_symbols = []
        found_symbols = []
        
        print(f"\n{Colors.BLUE}🔍 Checking USDT pair availability...{Colors.END}")
        
        for coin in all_coins:
            symbol = coin['symbol']
            
            if self.validate_symbol_exists(symbol):
                found_symbols.append(f"{symbol} → {symbol}USDT ✓")
            else:
                missing_symbols.append(f"{symbol} (no {symbol}USDT pair found)")
            
            time.sleep(0.1)  # Small delay for rate limiting
        
        success = len(missing_symbols) == 0
        
        if success:
            details = f"All symbols available as USDT pairs:\n" + '\n'.join([f"  • {s}" for s in found_symbols])
        else:
            details = f"Missing USDT pairs:\n" + '\n'.join([f"  • {s}" for s in missing_symbols])
            if found_symbols:
                details += f"\n\nFound USDT pairs:\n" + '\n'.join([f"  • {s}" for s in found_symbols])
        
        self.print_test_result("USDT Pair Availability", success, details)
        return success
    
    def validate_symbol_exists(self, symbol: str) -> bool:
        """Check if USDT pair exists for the symbol."""
        try:
            response = requests.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                params={'symbol': f"{symbol}USDT"},
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def test_volume_requirements(self, metadata: Dict[str, Any]) -> bool:
        """
        Test 6: Validate volume requirements using historical data from fixed date range.
        """
        targets = metadata.get('targets', [])
        anchors = metadata.get('anchors', [])
        
        print(f"\n{Colors.BLUE}📊 Analyzing volume data ({self.volume_start_date} to {self.volume_end_date})...{Colors.END}")
        print(f"\n{Colors.CYAN}Checking Targets (min ${self.min_target_volume_usd:,.0f} daily volume):{Colors.END}")
        
        issues = []
        valid_coins = []
        
        # Check target volume requirements
        for target in targets:
            symbol = target['symbol']
            print(f"\n  {Colors.WHITE}• Checking {symbol}...{Colors.END}")
            
            if not self.validate_symbol_exists(symbol):
                issues.append(f"Target {symbol}: USDT pair not available")
                print(f"    {Colors.RED}✗ USDT pair not available{Colors.END}")
                continue
            
            avg_volume = self.fetch_historical_volume_data(symbol)
            
            if avg_volume == 0.0:
                issues.append(f"Target {symbol}: Could not retrieve volume data")
                print(f"    {Colors.RED}✗ Could not retrieve volume data{Colors.END}")
            elif avg_volume < self.min_target_volume_usd:
                issues.append(
                    f"Target {symbol}: ${avg_volume:,.0f} < ${self.min_target_volume_usd:,.0f} (daily avg)"
                )
                print(f"    {Colors.RED}✗ Volume too low: ${avg_volume:,.0f} < ${self.min_target_volume_usd:,.0f}{Colors.END}")
            else:
                valid_coins.append(f"Target {symbol}: ${avg_volume:,.0f} ✓ (daily avg)")
                print(f"    {Colors.GREEN}✓ Volume sufficient: ${avg_volume:,.0f}{Colors.END}")
            
            time.sleep(0.1)  # Rate limiting
        
        print(f"\n{Colors.CYAN}Checking Anchors (min ${self.min_anchor_volume_usd:,.0f} daily volume):{Colors.END}")
        
        # Check anchor volume requirements  
        for anchor in anchors:
            symbol = anchor['symbol']
            print(f"\n  {Colors.WHITE}• Checking {symbol}...{Colors.END}")
            
            if not self.validate_symbol_exists(symbol):
                issues.append(f"Anchor {symbol}: USDT pair not available")
                print(f"    {Colors.RED}✗ USDT pair not available{Colors.END}")
                continue
            
            avg_volume = self.fetch_historical_volume_data(symbol)
            
            if avg_volume == 0.0:
                issues.append(f"Anchor {symbol}: Could not retrieve volume data")
                print(f"    {Colors.RED}✗ Could not retrieve volume data{Colors.END}")
            elif avg_volume < self.min_anchor_volume_usd:
                issues.append(
                    f"Anchor {symbol}: ${avg_volume:,.0f} < ${self.min_anchor_volume_usd:,.0f} (daily avg)"
                )
                print(f"    {Colors.RED}✗ Volume too low: ${avg_volume:,.0f} < ${self.min_anchor_volume_usd:,.0f}{Colors.END}")
            else:
                valid_coins.append(f"Anchor {symbol}: ${avg_volume:,.0f} ✓ (daily avg)")
                print(f"    {Colors.GREEN}✓ Volume sufficient: ${avg_volume:,.0f}{Colors.END}")
            
            time.sleep(0.1)  # Rate limiting
        
        success = len(issues) == 0
        
        if success:
            details = (f"All volume requirements met using daily averages:\n" + 
                      '\n'.join([f"  • {coin}" for coin in valid_coins]))
        else:
            details = (f"Volume requirement failures:\n" + 
                      '\n'.join([f"  • {issue}" for issue in issues]))
            if valid_coins:
                details += "\n\nPassing coins:\n" + '\n'.join([f"  • {coin}" for coin in valid_coins])
        
        self.print_test_result("Volume Requirements (Historical Average)", success, details)
        return success
    
    def test_generate_signals_function(self, metadata: Dict[str, Any]) -> bool:
        """
        Test 7: Check if generate_signals function exists and has correct signature.
        """
        try:
            if not hasattr(self.strategy_module, 'generate_signals'):
                self.print_test_result(
                    "Generate Signals Function", 
                    False, 
                    "generate_signals() function not found in strategy module"
                )
                return False
            
            func = getattr(self.strategy_module, 'generate_signals')
            if not callable(func):
                self.print_test_result(
                    "Generate Signals Function", 
                    False, 
                    "generate_signals exists but is not callable"
                )
                return False
            
            # Check function signature (basic inspection)
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            if len(params) < 2:
                self.print_test_result(
                    "Generate Signals Function", 
                    False, 
                    f"generate_signals should take 2 parameters (anchor_df, target_df), found: {params}"
                )
                return False
            
            self.print_test_result(
                "Generate Signals Function", 
                True, 
                f"Function found with parameters: {params}"
            )
            return True
            
        except Exception as e:
            self.print_test_result(
                "Generate Signals Function", 
                False, 
                f"Error checking function: {str(e)}"
            )
            return False
    
    def test_strategy_data_generation(self, metadata: Dict[str, Any]) -> Tuple[bool, pd.DataFrame]:
        """
        Test 8: Generate strategy data and test signal generation.
        """
        try:
            print(f"\n{Colors.BLUE}📊 Generating strategy data...{Colors.END}")
            
            # Initialize data manager
            data_manager = CryptoDataManager()
            
            # Convert metadata to required format for data manager
            all_symbol_configs = []
            
            # Add targets
            for target in metadata.get('targets', []):
                all_symbol_configs.append({
                    "symbol": target['symbol'],
                    "timeframe": target['timeframe']
                })
            
            # Add anchors
            for anchor in metadata.get('anchors', []):
                all_symbol_configs.append({
                    "symbol": anchor['symbol'], 
                    "timeframe": anchor['timeframe']
                })
            
            # Get the complete dataset
            full_df = data_manager.get_market_data(all_symbol_configs)
            
            if full_df.empty:
                self.print_test_result(
                    "Strategy Data Generation", 
                    False, 
                    "No data could be retrieved for strategy symbols"
                )
                return False, pd.DataFrame()
            
            # Separate anchor and target data
            anchor_columns = ['timestamp']
            target_columns = ['timestamp']
            
            for anchor in metadata.get('anchors', []):
                symbol = anchor['symbol']
                timeframe = anchor['timeframe']
                for col_type in ['open', 'high', 'low', 'close', 'volume']:
                    col_name = f"{col_type}_{symbol}_{timeframe}"
                    if col_name in full_df.columns:
                        anchor_columns.append(col_name)
            
            for target in metadata.get('targets', []):
                symbol = target['symbol']
                timeframe = target['timeframe']
                for col_type in ['open', 'high', 'low', 'close', 'volume']:
                    col_name = f"{col_type}_{symbol}_{timeframe}"
                    if col_name in full_df.columns:
                        target_columns.append(col_name)
            
            anchor_df = full_df[anchor_columns].copy()
            target_df = full_df[target_columns].copy()
            
            print(f"   Anchor data shape: {anchor_df.shape}")
            print(f"   Target data shape: {target_df.shape}")
            
            # Test signal generation
            generate_signals_func = getattr(self.strategy_module, 'generate_signals')
            signals_df = generate_signals_func(anchor_df, target_df)
            
            if not isinstance(signals_df, pd.DataFrame):
                self.print_test_result(
                    "Strategy Data Generation", 
                    False, 
                    f"generate_signals should return DataFrame, got {type(signals_df).__name__}"
                )
                return False, pd.DataFrame()
            
            print(f"   Generated signals shape: {signals_df.shape}")
            
            self.print_test_result(
                "Strategy Data Generation", 
                True, 
                f"Successfully generated data - Full: {full_df.shape}, Signals: {signals_df.shape}"
            )
            
            return True, signals_df
            
        except Exception as e:
            self.print_test_result(
                "Strategy Data Generation", 
                False, 
                f"Error during data generation: {str(e)}"
            )
            return False, pd.DataFrame()
    
    def test_signals_validation(self, signals_df: pd.DataFrame, metadata: Dict[str, Any]) -> bool:
        """
        Test 9: Validate the generated signals DataFrame structure and content.
        """
        if signals_df.empty:
            self.print_test_result(
                "Signals Validation", 
                False, 
                "No signals DataFrame provided (previous test failed)"
            )
            return False
        
        issues = []
        valid_checks = []
        
        # Check required columns
        required_columns = ["timestamp", "symbol", "signal", "position_size"]
        missing_columns = [col for col in required_columns if col not in signals_df.columns]
        
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        else:
            valid_checks.append(f"All required columns present: {required_columns}")
        
        # Check timestamp completeness (should have 1H frequency data)
        if 'timestamp' in signals_df.columns:
            unique_timestamps = signals_df['timestamp'].nunique()
            if unique_timestamps != self.expected_1h_rows:
                issues.append(
                    f"Timestamp count mismatch: {unique_timestamps} != {self.expected_1h_rows} (expected 1H frequency)"
                )
            else:
                valid_checks.append(f"Timestamp frequency correct: {unique_timestamps} rows (1H frequency)")
        
        # Check signal values
        if 'signal' in signals_df.columns:
            valid_signals = {'BUY', 'HOLD', 'SELL'}
            unique_signals = set(signals_df['signal'].unique())
            invalid_signals = unique_signals - valid_signals
            
            if invalid_signals:
                issues.append(f"Invalid signal values: {invalid_signals}. Must be one of: {valid_signals}")
            else:
                valid_checks.append(f"All signals valid: {unique_signals}")
        
        # Check position_size values
        if 'position_size' in signals_df.columns:
            position_sizes = signals_df['position_size']
            
            # Check if all values are between 0 and 1
            invalid_positions = position_sizes[(position_sizes < 0) | (position_sizes > 1)]
            if len(invalid_positions) > 0:
                issues.append(f"Invalid position sizes found: {len(invalid_positions)} values outside [0, 1] range")
            else:
                valid_checks.append("All position sizes within [0, 1] range")
            
            # Check if values are numeric (decimals allowed)
            non_numeric = position_sizes[pd.isna(pd.to_numeric(position_sizes, errors='coerce'))]
            if len(non_numeric) > 0:
                issues.append(f"Non-numeric position sizes found: {len(non_numeric)} values")
            else:
                valid_checks.append("All position sizes are numeric")
        
        # Check symbol values match strategy targets
        if 'symbol' in signals_df.columns:
            strategy_symbols = {target['symbol'] for target in metadata.get('targets', [])}
            signal_symbols = set(signals_df['symbol'].unique())
            
            unexpected_symbols = signal_symbols - strategy_symbols
            missing_symbols = strategy_symbols - signal_symbols
            
            if unexpected_symbols:
                issues.append(f"Unexpected symbols in signals: {unexpected_symbols}")
            if missing_symbols:
                issues.append(f"Missing symbols from signals: {missing_symbols}")
            
            if not unexpected_symbols and not missing_symbols:
                valid_checks.append(f"Symbol alignment correct: {signal_symbols}")
        
        # Check for minimum trading activity (buy-sell pairs)
        if 'signal' in signals_df.columns and 'symbol' in signals_df.columns:
            trading_activity_issues = []
            trading_activity_valid = []
            
            for symbol in signals_df['symbol'].unique():
                symbol_signals = signals_df[signals_df['symbol'] == symbol]['signal']
                
                buy_count = (symbol_signals == 'BUY').sum()
                sell_count = (symbol_signals == 'SELL').sum()
                
                # Check for minimum buy-sell pairs
                min_pairs = min(buy_count, sell_count)
                
                if min_pairs < 2:
                    trading_activity_issues.append(
                        f"{symbol}: Only {min_pairs} complete buy-sell pairs (BUY: {buy_count}, SELL: {sell_count})"
                    )
                else:
                    trading_activity_valid.append(
                        f"{symbol}: {min_pairs} buy-sell pairs (BUY: {buy_count}, SELL: {sell_count})"
                    )
            
            if trading_activity_issues:
                issues.extend([f"Insufficient trading activity - need minimum 2 buy-sell pairs per symbol:"] + 
                             [f"  {issue}" for issue in trading_activity_issues])
            else:
                valid_checks.extend([f"Sufficient trading activity:"] + 
                                  [f"  {activity}" for activity in trading_activity_valid])
        
        # Additional suggestions
        suggestions = []
        if 'position_size' in signals_df.columns:
            zero_positions = (signals_df['position_size'] == 0).sum()
            total_rows = len(signals_df)
            zero_percentage = (zero_positions / total_rows) * 100
            
            if zero_percentage > 80:
                suggestions.append(f"High percentage of zero positions: {zero_percentage:.1f}% - consider strategy activity")
        
        success = len(issues) == 0
        
        if success:
            details = "All signal validations passed:\n" + '\n'.join([f"  ✓ {check}" for check in valid_checks])
            if suggestions:
                details += "\n\nSuggestions:\n" + '\n'.join([f"  💡 {sug}" for sug in suggestions])
        else:
            details = "Signal validation failures:\n" + '\n'.join([f"  ✗ {issue}" for issue in issues])
            if valid_checks:
                details += "\n\nPassing checks:\n" + '\n'.join([f"  ✓ {check}" for check in valid_checks])
        
        self.print_test_result("Signals Validation", success, details)
        return success
    
    def run_all_tests(self) -> bool:
        """
        Run the complete validation suite.
        
        Executes all tests in sequence and returns overall success status.
        """
        self.print_header()
        
        # Test 1: Module Loading
        if not self.test_module_loading():
            return False
        
        # Test 2: Function Existence
        if not self.test_function_exists():
            return False
        
        # Test 3: Function Output Format
        success, metadata = self.test_function_output_format()
        if not success:
            return False
        
        # Test 4: Limits Compliance
        if not self.test_limits_compliance(metadata):
            return False
        
        # Test 5: Symbol Availability
        if not self.test_symbol_availability(metadata):
            return False
        
        # Test 6: Volume Requirements (using historical data)
        if not self.test_volume_requirements(metadata):
            return False
        
        # Test 7: Generate Signals Function
        if not self.test_generate_signals_function(metadata):
            return False
        
        # Test 8 & 9: Strategy Data Generation and Signals Validation
        success, signals_df = self.test_strategy_data_generation(metadata)
        if not success:
            return False
        
        if not self.test_signals_validation(signals_df, metadata):
            return False
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"{Colors.BOLD}{Colors.GREEN}🎉 ALL TESTS PASSED!{Colors.END}")
        print(f"{Colors.GREEN}Strategy is ready for deployment.{Colors.END}")
        print(f"{'='*60}\n")
        
        return True

def main():
    """Main entry point for the validation script."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Hardcoded strategy file path in the same directory
    strategy_path = os.path.join(script_dir, "strategy.py")
    
    if not os.path.exists(strategy_path):
        print(f"{Colors.RED}Error: Strategy file '{strategy_path}' not found{Colors.END}")
        sys.exit(1)
    
    validator = StrategyValidator(strategy_path)
    
    try:
        success = validator.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {str(e)}{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()