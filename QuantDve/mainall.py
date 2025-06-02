#This is like almost it, everything is perfect (almost) 1/4

#THIS IS IT!!! MAYBE MORE LSTM 
#it was done, for now...
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import traceback
import random
from collections import deque
from datetime import datetime, timedelta
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from hmmlearn import hmm
import scipy.stats as stats
import tensorflow
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, GRU, BatchNormalization, Bidirectional, Conv1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
from bloomberg import get_normalized_news_sentiment

TF_ENABLE_ONEDNN_OPTS=0

logging.getLogger("matplotlib").setLevel(logging.WARNING)

warnings.filterwarnings('ignore')

# Enable TensorFlow GPU support and log device placement
physical_devices = tensorflow.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"[INFO] Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        tensorflow.config.experimental.set_memory_growth(device, True)
    print("[INFO] GPU memory growth enabled")
else:
    print("[WARNING] No GPU found, using CPU instead")

# Alpha Vantage API key
#ALPHA_VANTAGE_API_KEY = "73KWO176IRABCOCJ"
#ALPHA_VANTAGE_API_KEY = "FAD56FMDCHAL8V1T"
#ALPHA_VANTAGE_API_KEY = "VEBT699FFQNZR85X"
ALPHA_VANTAGE_API_KEY = "IE4EDZ78E3WEUZKO"


# Output file
OUTPUT_FILE = "STOCK_ANALYSIS_RESULTS.txt"

from alpha_vantage_client import AlphaVantageClient

# Modified analyze_stock function (originally from semibloom.txt)
# Make sure the import 'from sentiment_analyzer import get_normalized_news_sentiment' is added at the top
def analyze_stock(symbol, client):
    """
    Analyze a stock and generate recommendations with improved prediction bands

    Parameters:
    -----------
    symbol: str
        Stock symbol to analyze
    client: AlphaVantageClient
        Alpha Vantage API client

    Returns:
    --------
    dict
        Analysis result with enhanced prediction bands or None on failure
    """
    try:
        # Fetch stock data
        stock_data = client.get_stock_data(symbol) #

        if stock_data is None or len(stock_data) < 60: #
            print(f"[WARNING] Insufficient data for {symbol}") #
            return None

        # Get company info and quote data
        company_info = client.get_company_overview(symbol) #
        quote_data = client.get_global_quote(symbol) #

        # Get current price
        current_price = quote_data['price'] if quote_data else stock_data['4. close'].iloc[-1] #

        try:
            # --- NEW: Fetch dynamic news sentiment ---
            print(f"[INFO] Fetching news sentiment for {symbol}...")
            sentiment_score = get_normalized_news_sentiment(symbol)
            # Use a default neutral score if fetching fails
            sentiment_score_for_calc = sentiment_score if sentiment_score is not None else 0.5
            if sentiment_score is not None:
                print(f"[INFO] Retrieved Normalized News Sentiment: {sentiment_score:.4f}")
            else:
                print(f"[WARNING] Could not retrieve news sentiment for {symbol}. Using default 0.5.")
        except ImportError:
            print("[WARNING] Error accesing bloomberg terminal. Proceeding with normal path...")
        except Exception as e:
            print(f"[ERROR] Error fetching data for {symbol}: {e}. Using Default 0.5")

        # --- MODIFIED: Pass sentiment_score_for_calc to calculate_sigma ---
        analysis_details = calculate_sigma(stock_data, sentiment_score_for_calc) # Pass the score here

        if analysis_details is None: #
            print(f"[WARNING] Failed to calculate sigma for {symbol}") #
            return None

        sigma = analysis_details["sigma"] #

        # Get recommendation
        recommendation = get_sigma_recommendation(sigma, analysis_details) #

        # Generate price predictions based on analysis
        predictions = None # Initialize predictions
        try: #
            # First try the optimized version with tighter bands
            predictions = generate_optimized_price_predictions(stock_data, analysis_details) #
            if predictions: #
                print(f"[INFO] Using optimized price predictions with tighter bands for {symbol}") #
        except Exception as pred_error: #
            print(f"[ERROR] Optimized prediction error: {pred_error}, using original method") #
            traceback.print_exc() #
            # Use original prediction function as fallback (assuming it exists)
            # predictions = generate_price_predictions(stock_data, analysis_details) #

        # Create prediction plot if predictions are available
        plot_path = None #
        if predictions: #
            try: #
                # Try using improved plotting first
                plot_path = create_improved_prediction_plot(stock_data, predictions, symbol) #
                if not plot_path: #
                    print("Error when creating advanced prediction plot.") #
                    time.sleep(1) #
                    print("Using normal prediction plot...") #
                    # plot_path = create_prediction_plot(stock_data, predictions, symbol) # Assuming this function exists
            except Exception as plot_error: #
                print(f"[ERROR] Improved plotting error: {plot_error}, using original method") #
                # Use original plotting function as fallback (assuming it exists)
                # plot_path = create_prediction_plot(stock_data, predictions, symbol) #

        # Run detailed drawdown analysis - enhanced with adaptive factors
        drawdown_analysis = analyze_drawdowns(stock_data, window_size=180) #
        if drawdown_analysis and isinstance(drawdown_analysis, dict): #
            # Add to analysis details if available
            analysis_details["drawdown_analysis"] = drawdown_analysis #
            # Extract key metrics for easy access
            analysis_details["max_drawdown"] = drawdown_analysis.get("max_drawdown", analysis_details.get("max_drawdown", 0)) #
            # Keep original risk_adjustment if available or use the new one
            if "risk_adjustment" not in analysis_details and "risk_adjustment" in drawdown_analysis: #
                analysis_details["risk_adjustment"] = drawdown_analysis.get("risk_adjustment", 1.0) #

        # Ensure we have risk metrics
        if "risk_metrics" not in analysis_details: #
            analysis_details["risk_metrics"] = {} #

        # Add drawdown probability and recovery metrics if available
        if drawdown_analysis: #
            if "drawdown_probability" in drawdown_analysis: #
                analysis_details["risk_metrics"]["drawdown_probability"] = drawdown_analysis["drawdown_probability"] #
            if "recovery_resilience" in drawdown_analysis: #
                analysis_details["risk_metrics"]["recovery_resilience"] = drawdown_analysis["recovery_resilience"] #

        # Calculate additional risk metrics and ratings
        risk_metrics = {} #
        # Include all existing risk metrics from analysis details
        if "sharpe" in analysis_details: #
            risk_metrics["sharpe"] = analysis_details["sharpe"] #
        if "kelly" in analysis_details: #
            risk_metrics["kelly"] = analysis_details["kelly"] #
        if "max_drawdown" in analysis_details: #
            risk_metrics["max_drawdown"] = analysis_details["max_drawdown"] #

        # Add drawdown analysis for risk assessment
        if drawdown_analysis: #
            risk_metrics["drawdown_analysis"] = drawdown_analysis #
            # Format key metrics for display
            risk_metrics["max_drawdown_pct"] = f"{drawdown_analysis.get('max_drawdown', 0)*100:.1f}%" #
            risk_metrics["drawdown_count"] = drawdown_analysis.get('drawdown_count', 0) #
            risk_metrics["avg_recovery_time"] = f"{drawdown_analysis.get('avg_recovery_time', 0):.0f} days" #
            # Calculate risk rating (1-5 scale)
            dd_severity = abs(drawdown_analysis.get('max_drawdown', 0)) * 5  # 0-50% → 0-2.5 #
            dd_frequency = min(1.0, drawdown_analysis.get('drawdown_frequency', 0) / 4)  # 0-4 → 0-1 #
            recovery_factor = 1.0 - min(1.0, drawdown_analysis.get('recovery_resilience', 0.5))  # 0-1 → 1-0 #
            risk_rating = min(5, max(1, round((dd_severity + dd_frequency + recovery_factor) * 1.67))) #
            risk_metrics["risk_rating"] = risk_rating #
            risk_metrics["risk_level"] = ["Very Low", "Low", "Moderate", "High", "Very High"][risk_rating-1] #

        # Calculate price targets more explicitly for easy access
        price_targets = {} #
        if predictions: #
            price_targets["price_target_30d"] = predictions["price_target_30d"] #
            price_targets["price_target_60d"] = predictions["price_target_60d"] #
            price_targets["expected_return_30d"] = f"{predictions['expected_return_30d']:.1f}%" #
            price_targets["expected_return_60d"] = f"{predictions['expected_return_60d']:.1f}%" #
            # Add band width metrics to assess prediction confidence
            try: #
                price_30d = predictions["price_target_30d"] #
                band_width_30d = (predictions["upper_bound_68"][29] - predictions["lower_bound_68"][29]) / price_30d #
                price_targets["band_width_30d"] = f"±{band_width_30d*50:.1f}%"  # Convert to ± percentage #
                price_60d = predictions["price_target_60d"] #
                band_width_60d = (predictions["upper_bound_68"][59] - predictions["lower_bound_68"][59]) / price_60d #
                price_targets["band_width_60d"] = f"±{band_width_60d*50:.1f}%"  # Convert to ± percentage #
                # Add upside and downside potential
                price_targets["upside_potential_30d"] = f"{((predictions['upper_bound_68'][29] / current_price) - 1) * 100:.1f}%" #
                price_targets["downside_risk_30d"] = f"{((predictions['lower_bound_68'][29] / current_price) - 1) * 100:.1f}%" #
            except (IndexError, KeyError) as e: #
                print(f"[WARNING] Could not calculate all band metrics: {e}") #

        # Create result dictionary with all components
        result = {
            "symbol": symbol, #
            "price": current_price, #
            "sigma": sigma, #
            "recommendation": recommendation, #
            "company_info": company_info, #
            "quote_data": quote_data, #
            "analysis": analysis_details, #
            "predictions": predictions, #
            "plot_path": plot_path, #
            "risk_metrics": risk_metrics, #
            "price_targets": price_targets, #
            # --- NEW: Add the actual sentiment score (or None) to the results ---
            "news_sentiment_score": sentiment_score
        }

        # Log summary of analysis
        print(f"[INFO] Analysis for {symbol} completed successfully.") #
        print(f"       Sigma: {sigma:.3f}, Current Price: ${current_price:.2f}") #
        # --- NEW: Also log sentiment score ---
        if sentiment_score is not None:
             print(f"       News Sentiment: {sentiment_score:.4f}")
        # --- End NEW ---
        if predictions: #
            print(f"       30-Day Target: ${predictions['price_target_30d']:.2f} ({predictions['expected_return_30d']:.1f}%)") #
            print(f"       60-Day Target: ${predictions['price_target_60d']:.2f} ({predictions['expected_return_60d']:.1f}%)") #

        return result
    except Exception as e: #
        print(f"[ERROR] Failed to analyze {symbol}: {e}") #
        traceback.print_exc() #
        return None

# 2. Second modification: Enhanced drawdown analysis with adaptive factors
def analyze_drawdowns(df, window_size=180):  # Changed from 60 to 180 to match LSTM
    """
    Advanced drawdown analysis tailored to individual stock characteristics
    
    Parameters:
    -----------
    df: pandas.DataFrame
        DataFrame with historical data including returns
    window_size: int
        Size of the sliding window for analyzing drawdown patterns
        
    Returns:
    --------
    dict
        Dictionary with detailed drawdown metrics
    """
    try:
        print(f"[INFO] Analyzing drawdown patterns with window size: {window_size} days (matching LSTM window)")
        
        # Use log returns if available for more accurate drawdown calculation
        if 'log_returns' in df.columns:
            returns = df['log_returns'].dropna()
            cumulative_returns = np.exp(np.cumsum(returns)) - 1
            print("[INFO] Using log returns for drawdown analysis")
        elif 'returns' in df.columns:
            returns = df['returns'].dropna()
            cumulative_returns = (1 + returns).cumprod() - 1
            print("[INFO] Using regular returns for drawdown analysis")
        else:
            # Fallback to calculating returns from price
            price_col = 'close' if 'close' in df.columns else '4. close'
            prices = df[price_col].dropna().values
            returns = pd.Series(np.diff(np.log(prices)))
            cumulative_returns = np.exp(np.cumsum(returns)) - 1
            print("[INFO] Calculated returns from price data for drawdown analysis")
        
        # Calculate drawdowns
        peak = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / peak) - 1
        
        # Identify drawdown events
        # A drawdown event begins when a drawdown starts and ends when we return to the previous peak
        drawdown_events = []
        in_drawdown = False
        start_idx = 0
        max_drawdown = 0
        
        for i, dd in enumerate(drawdowns):
            if not in_drawdown and dd < -0.02:  # Start of a significant drawdown (> 2%)
                in_drawdown = True
                start_idx = i
                max_drawdown = dd
            elif in_drawdown:
                if dd < max_drawdown:  # Drawdown getting deeper
                    max_drawdown = dd
                if dd >= 0:  # Recovered to previous peak
                    # Record the drawdown event
                    duration = i - start_idx
                    drawdown_events.append({
                        'start_idx': start_idx,
                        'end_idx': i,
                        'duration': duration,
                        'depth': max_drawdown,
                        'recovery_time': duration,  # Full recovery time
                        'avg_recovery_rate': -max_drawdown / duration if duration > 0 else 0
                    })
                    in_drawdown = False
        
        # If still in drawdown at the end of the data
        if in_drawdown:
            duration = len(drawdowns) - start_idx
            drawdown_events.append({
                'start_idx': start_idx,
                'end_idx': len(drawdowns) - 1,
                'duration': duration,
                'depth': max_drawdown,
                'recovery_time': None,  # Hasn't recovered yet
                'avg_recovery_rate': None
            })
        
        # Analyze drawdown statistics
        if not drawdown_events:
            print("[INFO] No significant drawdown events detected")
            return {
                'max_drawdown': drawdowns.min(),
                'drawdown_count': 0,
                'avg_drawdown_depth': 0,
                'avg_drawdown_duration': 0,
                'recovery_rate': 0,
                'drawdown_frequency': 0,
                'drawdown_severity': 0,
                'risk_adjustment': 1.0
            }
        
        # Calculate statistics
        drawdown_depths = [event['depth'] for event in drawdown_events]
        drawdown_durations = [event['duration'] for event in drawdown_events]
        
        # Filter for completed drawdowns to calculate recovery stats
        completed_events = [event for event in drawdown_events if event['recovery_time'] is not None]
        recovery_times = [event['recovery_time'] for event in completed_events] if completed_events else [0]
        recovery_rates = [event['avg_recovery_rate'] for event in completed_events if event['avg_recovery_rate'] is not None]
        
        # Calculate average statistics
        avg_depth = np.mean(drawdown_depths)
        avg_duration = np.mean(drawdown_durations)
        avg_recovery_time = np.mean(recovery_times) if recovery_times else None
        avg_recovery_rate = np.mean(recovery_rates) if recovery_rates else None
        
        # Calculate drawdown frequency (events per year)
        total_trading_days = len(returns)
        years = total_trading_days / 252  # Assuming 252 trading days per year
        drawdown_frequency = len(drawdown_events) / years
        
        # Calculate drawdown severity - combined measure of depth and duration
        # More negative (deeper) drawdowns and longer durations = higher severity
        drawdown_severity = np.mean([-depth * duration for depth, duration in zip(drawdown_depths, drawdown_durations)])
        
        # Calculate probability of future drawdowns based on historical pattern
        # More frequent and severe drawdowns increase this probability
        # Normalized to a 0-1 scale where 1 = highest probability
        drawdown_probability = min(1.0, (drawdown_frequency / 4.0) * (drawdown_severity / 0.2))
        
        # Calculate recovery resilience - how quickly the stock tends to recover
        # Normalized to a 0-1 scale where 1 = most resilient (faster recovery)
        if completed_events and avg_recovery_rate:
            recovery_resilience = min(1.0, max(0.0, 0.5 - avg_recovery_rate * 100))
        else:
            recovery_resilience = 0.5  # Neutral if no data
        
        # Calculate risk adjustment factor for sigma
        # This will adapt based on the specific drawdown characteristics of the stock
        # More severe/frequent drawdowns = higher discount (lower factor)
        # More resilient recovery = less discount (higher factor)
        risk_adjustment = calculate_adaptive_risk_adjustment(
            max_drawdown=min(drawdown_depths),
            drawdown_frequency=drawdown_frequency,
            drawdown_severity=drawdown_severity,
            recovery_resilience=recovery_resilience
        )
        
        # Return comprehensive drawdown analysis
        return {
            'max_drawdown': min(drawdown_depths),
            'drawdown_count': len(drawdown_events),
            'avg_drawdown_depth': avg_depth,
            'avg_drawdown_duration': avg_duration,
            'avg_recovery_time': avg_recovery_time,
            'avg_recovery_rate': avg_recovery_rate,
            'drawdown_frequency': drawdown_frequency,
            'drawdown_severity': drawdown_severity,
            'drawdown_probability': drawdown_probability,
            'recovery_resilience': recovery_resilience,
            'risk_adjustment': risk_adjustment,
            'drawdown_events': drawdown_events
        }
        
    except Exception as e:
        print(f"[ERROR] Error in drawdown analysis: {e}")
        traceback.print_exc()
        return {
            'max_drawdown': drawdowns.min() if 'drawdowns' in locals() else 0,
            'error': str(e),
            'risk_adjustment': 0.8  # Default conservative adjustment on error
        }

import numpy as np
import time
import traceback
# Modified calculate_sigma function (originally from semibloom.txt)
# Note the added 'sentiment_score' argument
# calculate_sigma function modified based on user request
def calculate_sigma(data, sentiment_score_input):
    """Calculate comprehensive sigma metric using log returns-based mean reversion with normalized sentiment (Modified: News sentiment adjustment applied post-calculation)"""
    try:
        # Set a maximum execution time for the entire function
        max_execution_time = 60000000  # 10 minutes max
        start_time = time.time()

        # 1. Calculate technical indicators with log returns mean reversion components
        indicators_df = calculate_technical_indicators(data)
        if indicators_df is None or len(indicators_df) < 30:
            print("[WARNING] Technical indicators calculation failed or insufficient data")
            return None

        # 2. Calculate Hurst exponent using log returns for more accurate results
        hurst_info = calculate_hurst_exponent(indicators_df, use_log_returns=True)
        print(f"[INFO] Hurst exponent: {hurst_info['hurst']:.3f} - {hurst_info['regime']}")

        # 3. Calculate mean reversion half-life using log returns
        half_life_info = calculate_mean_reversion_half_life(indicators_df)
        print(
            f"[INFO] Mean reversion half-life: {half_life_info['half_life']:.1f} days - {half_life_info['mean_reversion_speed']} (beta: {half_life_info.get('beta', 0):.3f})")

        # 4. Analyze volatility regimes with log returns
        vol_data = analyze_volatility_regimes(indicators_df)
        print(
            f"[INFO] Volatility regime: {vol_data['vol_regime']} (Term structure: {vol_data['vol_term_structure']:.2f}, Persistence: {vol_data.get('vol_persistence', 0):.2f})")

        # 5. Detect market regime with log returns
        market_regime = detect_market_regime(indicators_df)
        print(
            f"[INFO] Market regime: {market_regime['current_regime']} (Duration: {market_regime['regime_duration']} days)")

        # 6. Apply PCA to reduce feature dimensionality
        pca_results = None
        pca_variance = []
        pca_components = None

        if time.time() - start_time < max_execution_time * 0.6:  # More generous allocation
            try:
                lookback_period = min(120, len(indicators_df))
                pca_results, pca_variance = apply_pca(indicators_df.iloc[-lookback_period:])
                if pca_results is not None:
                    pca_components = pca_results.iloc[-1].values
                    print(f"[DEBUG] PCA components for latest datapoint: {pca_components}")
            except Exception as e:
                print(f"[WARNING] PCA calculation failed: {e}, continuing without it")
                pca_variance = []
        else:
            print("[WARNING] Skipping PCA calculation due to significant time constraints")

        # 7. Get LSTM volatility prediction with log returns features
        lstm_prediction = 0
        if time.time() - start_time < max_execution_time * 0.7:
            lstm_prediction = predict_with_lstm(data)
            print(f"[DEBUG] LSTM prediction: {lstm_prediction}")
        else:
            print("[WARNING] Skipping LSTM prediction due to time constraints")

        # 8. Get DQN recommendation with log returns features
        dqn_recommendation = 0.5  # Default neutral
        if time.time() - start_time < max_execution_time * 0.8:
            dqn_recommendation = get_dqn_recommendation(indicators_df)
            print(f"[DEBUG] DQN recommendation: {dqn_recommendation}")
        else:
            print("[WARNING] Skipping DQN recommendation due to time constraints")

        # Get latest technical indicators
        latest = indicators_df.iloc[-1]

        # MOMENTUM INDICATORS
        traditional_volatility = indicators_df['log_volatility'].iloc[-1] if 'log_volatility' in indicators_df.columns else indicators_df['volatility'].iloc[-1] if 'volatility' in indicators_df.columns else 0

        rsi = latest['RSI'] if not np.isnan(latest['RSI']) else 50
        rsi_signal = (max(0, min(100, rsi)) - 30) / 70
        rsi_signal = max(0, min(1, rsi_signal))

        macd = latest['MACD'] if not np.isnan(latest['MACD']) else 0
        macd_signal = np.tanh(macd * 10)

        sma20 = latest['SMA20'] if not np.isnan(latest['SMA20']) else 1
        sma50 = latest['SMA50'] if not np.isnan(latest['SMA50']) else 1
        sma_trend = (sma20 / sma50 - 1) if abs(sma50) > 1e-6 else 0
        sma_signal = np.tanh(sma_trend * 10)

        # Calculate short-term momentum (last 10 days vs previous 10 days)
        try:
            if 'log_returns' in indicators_df.columns:
                recent_returns_m = indicators_df['log_returns'].iloc[-10:].mean()
                previous_returns_m = indicators_df['log_returns'].iloc[-20:-10].mean()
                print("[INFO] Using log returns for momentum calculation")
            else:
                recent_returns_m = indicators_df['returns'].iloc[-10:].mean()
                previous_returns_m = indicators_df['returns'].iloc[-20:-10].mean()

            momentum_signal = np.tanh((recent_returns_m - previous_returns_m) * 20)
            momentum_signal = (momentum_signal + 1) / 2  # Convert to 0-1 scale
        except:
            momentum_signal = 0.5  # Neutral

        # MEAN REVERSION INDICATORS
        dist_from_sma200 = latest['dist_from_SMA200'] if not np.isnan(latest['dist_from_SMA200']) else 0
        sma200_signal = 1 - min(1, max(0, (dist_from_sma200 + 0.1) / 0.2))

        if 'log_returns_zscore' in latest and not np.isnan(latest['log_returns_zscore']):
            log_z = latest['log_returns_zscore']
            log_z_signal = min(1, max(0, (abs(log_z) - 0.5) / 2.5))
            print(f"[INFO] Using log returns z-score for mean reversion signal: {log_z:.2f} → {log_z_signal:.2f}")
            bb_reversal_signal = log_z_signal
        elif 'BB_pctB' in latest and not np.isnan(latest['BB_pctB']):
            bb_pctb = latest['BB_pctB']
            bb_reversal_signal = 1 - 2 * abs(bb_pctb - 0.5)
            bb_reversal_signal = max(0, min(1, bb_reversal_signal + 0.5))
            print(f"[INFO] Using Bollinger Band %B for mean reversion signal: {bb_pctb:.2f} → {bb_reversal_signal:.2f}")
        else:
            bb_reversal_signal = 0.5

        if 'log_expected_reversion_pct' in latest and not np.isnan(latest['log_expected_reversion_pct']):
            exp_rev = latest['log_expected_reversion_pct']
            accel_signal = min(1, abs(exp_rev) / 10)
            print(f"[INFO] Using log-based expected reversion: {exp_rev:.2f}% → {accel_signal:.2f}")
        elif 'ROC_accel' in latest and not np.isnan(latest['ROC_accel']):
            roc_accel = latest['ROC_accel']
            accel_signal = max(0, min(1, 0.5 - roc_accel * 10))
            print(f"[INFO] Using ROC acceleration: {roc_accel:.4f} → {accel_signal:.2f}")
        else:
            accel_signal = 0.5

        if 'log_mr_potential' in latest and not np.isnan(latest['log_mr_potential']):
            log_mr = latest['log_mr_potential']
            mean_rev_signal = min(1, abs(log_mr) / 2)
            print(f"[INFO] Using log-based mean reversion potential: {log_mr:.2f} → {mean_rev_signal:.2f}")
        elif 'mean_reversion_z' in latest and not np.isnan(latest['mean_reversion_z']):
            mean_rev_z = latest['mean_reversion_z']
            mean_rev_signal = min(1, abs(mean_rev_z) / 2)
            print(f"[INFO] Using regular mean reversion z-score: {mean_rev_z:.2f} → {mean_rev_signal:.2f}")
        else:
            mean_rev_signal = 0.5

        rsi_div = latest['rsi_divergence'] if 'rsi_divergence' in latest and not np.isnan(latest['rsi_divergence']) else 0
        rsi_div_signal = 1 if rsi_div < 0 else 0

        if 'log_autocorr_5' in latest and not np.isnan(latest['log_autocorr_5']):
            log_autocorr = latest['log_autocorr_5']
            overbought_signal = max(0, min(1, 0.5 - log_autocorr))
            print(f"[INFO] Using log returns autocorrelation: {log_autocorr:.2f} → {overbought_signal:.2f}")
        elif 'returns_zscore_20' in latest and not np.isnan(latest['returns_zscore_20']):
            returns_z = latest['returns_zscore_20']
            overbought_signal = max(0, min(1, (returns_z + 1) / 4))
            print(f"[INFO] Using returns z-score: {returns_z:.2f} → {overbought_signal:.2f}")
        else:
            overbought_signal = 0.5

        if 'log_vol_ratio' in latest and not np.isnan(latest['log_vol_ratio']):
            log_vol_ratio = latest['log_vol_ratio']
            vol_increase_signal = max(0, min(1, (log_vol_ratio - 0.8) / 1.2))
            print(f"[INFO] Using log volatility ratio: {log_vol_ratio:.2f} → {vol_increase_signal:.2f}")
        elif 'vol_ratio' in latest and not np.isnan(latest['vol_ratio']):
            vol_ratio = latest['vol_ratio']
            vol_increase_signal = max(0, min(1, (vol_ratio - 0.8) / 1.2))
            print(f"[INFO] Using volatility ratio: {vol_ratio:.2f} → {vol_increase_signal:.2f}")
        else:
            vol_increase_signal = 0.5

        williams_r = (latest['Williams_%R'] + 100) / 100 if 'Williams_%R' in latest and not np.isnan(latest['Williams_%R']) else 0.5
        cmf = (latest['CMF'] + 1) / 2 if 'CMF' in latest and not np.isnan(latest['CMF']) else 0.5

        # Assume normalized_sentiment is imported from bloomberg.py
        # Retrieve news sentiment but don't include it in the initial component calculation
        news_sentiment_value = sentiment_score_input
        print(f"[INFO] Using normalized news sentiment (for adjustment): {news_sentiment_value:.2f}")

        # Component groups for Sigma calculation
        momentum_components = {
            "rsi": rsi_signal,
            "macd": (macd_signal + 1) / 2,
            "sma_trend": (sma_signal + 1) / 2,
            "traditional_volatility": min(1, traditional_volatility * 25),
            "momentum": momentum_signal,
            "williams_r": williams_r,
            "cmf": cmf,
            "lstm": lstm_prediction,
            "dqn": dqn_recommendation
            # "news_sentiment": news_sentiment_value # REMOVED from here
        }

        reversion_components = {
            "sma200_signal": sma200_signal,
            "bb_reversal": bb_reversal_signal,
            "accel_signal": accel_signal,
            "mean_rev_signal": mean_rev_signal, # This was missing in original reversion score calc
            "rsi_div_signal": rsi_div_signal,
            "overbought_signal": overbought_signal,
            "vol_increase_signal": vol_increase_signal
        }

        print(f"[DEBUG] Momentum components: {momentum_components}")
        print(f"[DEBUG] Mean reversion components: {reversion_components}")

        # Calculate base momentum score (bullish when high) - renormalized weights
        # Original total weight in 'if' was 1.0. Without news (0.13), it's 0.87.
        # Original total weight in 'else' was 1.0. Without news (0.05), it's 0.95.

        if lstm_prediction > 0 and dqn_recommendation != 0.500:
            base_momentum_score = (
                (0.14 / 0.87) * momentum_components["traditional_volatility"] + # ≈ 0.161
                (0.07 / 0.87) * momentum_components["rsi"] +                   # ≈ 0.080
                (0.06 / 0.87) * momentum_components["macd"] +                   # ≈ 0.069
                (0.06 / 0.87) * momentum_components["sma_trend"] +              # ≈ 0.069
                (0.08 / 0.87) * momentum_components["momentum"] +               # ≈ 0.092
                (0.08 / 0.87) * momentum_components["williams_r"] +             # ≈ 0.092
                (0.08 / 0.87) * momentum_components["cmf"] +                    # ≈ 0.092
                (0.15 / 0.87) * momentum_components["lstm"] +                   # ≈ 0.172
                (0.15 / 0.87) * momentum_components["dqn"]                      # ≈ 0.172
            )
        else:
            base_momentum_score = (
                (0.20 / 0.95) * momentum_components["traditional_volatility"] + # ≈ 0.211
                (0.15 / 0.95) * momentum_components["rsi"] +                   # ≈ 0.158
                (0.15 / 0.95) * momentum_components["macd"] +                   # ≈ 0.158
                (0.15 / 0.95) * momentum_components["sma_trend"] +              # ≈ 0.158
                (0.15 / 0.95) * momentum_components["momentum"] +               # ≈ 0.158
                (0.10 / 0.95) * momentum_components["williams_r"] +             # ≈ 0.105
                (0.05 / 0.95) * momentum_components["cmf"]                      # ≈ 0.053
            )

        # Ensure base score is valid before adjustment
        base_momentum_score = np.nan_to_num(base_momentum_score, nan=0.5) # Handle potential NaNs
        base_momentum_score = max(0.0, min(1.0, base_momentum_score)) # Clamp base score

        # --- Apply News Sentiment Adjustment ---
        # Adjustment Logic (Example: Conditional Boost/Penalty)
        positive_threshold = 0.7  # Tune this threshold
        negative_threshold = 0.3  # Tune this threshold
        adjustment_amount = 0.05 # Tune this amount - how much news impacts the score

        final_momentum_score = base_momentum_score # Start with the base technical/model score

        if news_sentiment_value > positive_threshold:
            final_momentum_score += adjustment_amount
            print(f"[INFO] Boosting momentum score by {adjustment_amount:.3f} due to positive news ({news_sentiment_value:.2f})")
        elif news_sentiment_value < negative_threshold:
            final_momentum_score -= adjustment_amount
            print(f"[INFO] Penalizing momentum score by {adjustment_amount:.3f} due to negative news ({news_sentiment_value:.2f})")

        # --- Final Step: Ensure Score is within [0, 1] range ---
        final_momentum_score = max(0.0, min(1.0, final_momentum_score))

        # Assign the adjusted score to the 'momentum_score' variable for later use
        momentum_score = final_momentum_score
        # --- End of News Adjustment Section ---


        # Calculate mean reversion score (bearish when high)
        # Note: Added the missing mean_rev_signal component based on structure
        reversion_score = (
            0.1 * reversion_components["sma200_signal"] +
            0.15 * reversion_components["bb_reversal"] +
            0.2 * reversion_components["accel_signal"] +
            0.15 * reversion_components["mean_rev_signal"] + # Added this component weight
            0.10 * reversion_components["rsi_div_signal"] +
            0.15 * reversion_components["overbought_signal"] +
            0.10 * reversion_components["vol_increase_signal"]
        )
        # Normalize weights to sum to 1 (0.1 + 0.15 + 0.2 + 0.15 + 0.10 + 0.15 + 0.10 = 0.95)
        # Divide each by 0.95 or adjust weights to sum to 1. Here we adjust:
        reversion_score = (
            0.10 * reversion_components["sma200_signal"] +       # Adjusted: 0.10 / 0.95 ≈ 0.105
            0.16 * reversion_components["bb_reversal"] +       # Adjusted: 0.15 / 0.95 ≈ 0.158
            0.21 * reversion_components["accel_signal"] +        # Adjusted: 0.20 / 0.95 ≈ 0.211
            0.16 * reversion_components["mean_rev_signal"] +     # Adjusted: 0.15 / 0.95 ≈ 0.158
            0.10 * reversion_components["rsi_div_signal"] +      # Adjusted: 0.10 / 0.95 ≈ 0.105
            0.16 * reversion_components["overbought_signal"] +   # Adjusted: 0.15 / 0.95 ≈ 0.158
            0.11 * reversion_components["vol_increase_signal"]   # Adjusted: 0.10 / 0.95 ≈ 0.105
        ) # Sum approx 1.0

        # Get recent monthly return using log returns if available
        if 'log_returns' in indicators_df.columns:
            recent_returns = indicators_df['log_returns'].iloc[-20:].sum()
            recent_returns = np.exp(recent_returns) - 1
            print(f"[INFO] Using accumulated log returns for monthly return: {recent_returns:.2%}")
        else:
            recent_returns = latest['ROC_20'] if 'ROC_20' in latest and not np.isnan(latest['ROC_20']) else 0
            print(f"[INFO] Using ROC_20 for monthly return: {recent_returns:.2%}")

        # Adjust balance factor based on Hurst exponent
        hurst_adjustment = 0
        if hurst_info['hurst'] < 0.4:
            hurst_adjustment = 0.05
        elif hurst_info['hurst'] < 0.45:
            hurst_adjustment = 0.1
        elif hurst_info['hurst'] > 0.65:
            hurst_adjustment = -0.10
        elif hurst_info['hurst'] > 0.55:
            hurst_adjustment = -0.05

        base_balance_factor = 0.5 + hurst_adjustment

        # Add adjustment based on mean reversion half-life and beta
        half_life = half_life_info.get('half_life', 0)
        beta = half_life_info.get('beta', 0)

        mr_speed_adjustment = 0
        if -1 < beta < -0.5:
            mr_speed_adjustment = 0.05
        elif -0.5 < beta < -0.2:
            mr_speed_adjustment = 0.025
        elif beta > 0.2:
            mr_speed_adjustment = -0.03

        if 0 < half_life < 10:
            mr_speed_adjustment += 0.03
        elif 10 <= half_life < 30:
            mr_speed_adjustment += 0.015

        base_balance_factor += mr_speed_adjustment
        print(f"[INFO] Mean reversion adjustment based on beta/half-life: {mr_speed_adjustment:.3f}")

        # --- REMOVED NEWS ADJUSTMENT FROM BALANCE FACTOR ---
        # Adjust based on news sentiment (extreme sentiment increases mean reversion weight)
        # sentiment_adjustment = 0
        # if news_sentiment_value > 0.8:  # Very positive sentiment
        #     sentiment_adjustment = 0.05  # Slightly more weight to mean reversion
        #     print(f"[INFO] Increasing mean reversion weight due to high news sentiment: {news_sentiment_value:.2f}")
        # elif news_sentiment_value < 0.2:  # Very negative sentiment
        #     sentiment_adjustment = 0.05  # Slightly more weight to mean reversion
        #     print(f"[INFO] Increasing mean reversion weight due to low news sentiment: {news_sentiment_value:.2f}")
        # base_balance_factor += sentiment_adjustment
        # --- END REMOVED SECTION ---


        # For stocks with recent large moves, increase the mean reversion weight
        if recent_returns > 0.15:
            excess_return_factor = min(0.3, (recent_returns - 0.15) * 2)
            balance_factor = base_balance_factor + excess_return_factor
            print(f"[INFO] Increasing mean reversion weight by {excess_return_factor:.2f} due to high recent returns ({recent_returns:.1%})")
        elif recent_returns < -0.15:
            balance_factor = max(0.3, base_balance_factor - 0.1)
            print(f"[INFO] Decreasing mean reversion weight due to significant recent decline ({recent_returns:.1%})")
        else:
            balance_factor = base_balance_factor

        # Adjust based on volatility regime
        if vol_data['vol_regime'] == "Rising":
            balance_factor += 0.05
            print("[INFO] Increasing mean reversion weight due to rising volatility regime")
        elif vol_data['vol_regime'] == "Falling":
            balance_factor -= 0.05
            print("[INFO] Decreasing mean reversion weight due to falling volatility regime")

        # Adjust based on volatility persistence
        vol_persistence = vol_data.get('vol_persistence', 0.8)
        if vol_persistence > 0.9:
            balance_factor += 0.05
            print(f"[INFO] Increasing mean reversion weight due to high volatility persistence: {vol_persistence:.2f}")
        elif vol_persistence < 0.7:
            balance_factor = (balance_factor + 0.5) / 2
            print(f"[INFO] Adjusting balance factor toward neutral due to low volatility persistence: {vol_persistence:.2f}")

        # Ensure balance factor is reasonable
        balance_factor = max(0.2, min(0.8, balance_factor))

        # Calculate final sigma with balanced approach
        ensemble_result = create_ensemble_prediction(
            momentum_score, # Use the news-adjusted momentum score
            reversion_score,
            lstm_prediction,
            dqn_recommendation,
            vol_data,
            market_regime,
            hurst_info,
            half_life_info
        )

        if ensemble_result and "ensemble_score" in ensemble_result:
            sigma = ensemble_result["ensemble_score"]
            weights = ensemble_result["weights"]
            print(f"[INFO] Using ensemble model with weights: {weights}")
        else:
            # Use news-adjusted momentum score here
            sigma = momentum_score * (1 - balance_factor) + (1 - reversion_score) * balance_factor

        # Add small PCA adjustment if available
        if pca_components is not None and len(pca_components) >= 3:
            pca_influence = np.tanh(np.sum(pca_components[:3]) / 3) * 0.05
            sigma += pca_influence
            print(f"[DEBUG] PCA adjustment to Sigma: {pca_influence:.3f}")

        # Calculate risk-adjusted metrics with log returns
        risk_metrics = calculate_risk_adjusted_metrics(indicators_df, sigma)

        # Use risk-adjusted sigma
        final_sigma = risk_metrics.get("risk_adjusted_sigma", sigma)

        # Ensure sigma is between 0 and 1
        final_sigma = max(0, min(1, final_sigma))

        print(
            f"[INFO] Final components: Momentum(Adj)={momentum_score:.3f}, Reversion={reversion_score:.3f}, Balance={balance_factor:.2f}, Base Sigma={sigma:.3f}, Final Sigma={final_sigma:.3f}"
        )

        # Analysis details
        analysis_details = {
            "sigma": final_sigma,
            "raw_sigma": sigma,
            "momentum_score": momentum_score, # Store the adjusted momentum score
            "reversion_score": reversion_score,
            "balance_factor": balance_factor,
            "recent_monthly_return": recent_returns,
            "traditional_volatility": traditional_volatility,
            "rsi": rsi,
            "macd": macd,
            "sma_trend": sma_trend,
            "dist_from_sma200": dist_from_sma200,
            "last_price": latest['4. close'] if not np.isnan(latest['4. close']) else 0,
            "lstm_prediction": lstm_prediction,
            "dqn_recommendation": dqn_recommendation,
            "news_sentiment": news_sentiment_value, # Store the raw news sentiment used for adjustment
            "hurst_exponent": hurst_info['hurst'],
            "hurst_regime": hurst_info['regime'],
            "mean_reversion_half_life": half_life_info['half_life'],
            "mean_reversion_speed": half_life_info['mean_reversion_speed'],
            "mean_reversion_beta": half_life_info.get('beta', 0),
            "volatility_regime": vol_data['vol_regime'],
            "vol_term_structure": vol_data['vol_term_structure'],
            "vol_persistence": vol_data.get('vol_persistence', 0.8),
            "market_regime": market_regime['current_regime'],
            "max_drawdown": risk_metrics.get("max_drawdown", 0),
            "kelly": risk_metrics.get("kelly", 0),
            "sharpe": risk_metrics.get("sharpe", 0)
        }
        # Merge risk metrics properly
        analysis_details.update(risk_metrics)

        return analysis_details
    except Exception as e:
        print(f"[ERROR] Error calculating balanced Sigma with log returns: {e}")
        traceback.print_exc()
        return None

# Enhanced technical indicators with log returns mean reversion components
def calculate_technical_indicators(data):
    try:
        print(f"[DEBUG] Calculating enhanced technical indicators with log returns on data with shape: {data.shape}")
        df = data.copy()

        # Check if data is sufficient
        if len(df) < 50:
            print("[WARNING] Not enough data for technical indicators calculation")
            return None

        # Calculate regular returns if not already present
        if 'returns' not in df.columns:
            df['returns'] = df['4. close'].pct_change()
            df['returns'] = df['returns'].fillna(0)

        # Calculate log returns if not already present
        if 'log_returns' not in df.columns:
            df['log_returns'] = np.log(df['4. close'] / df['4. close'].shift(1))
            df['log_returns'] = df['log_returns'].fillna(0)

        # Calculate volatility (20-day rolling standard deviation)
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volatility'] = df['volatility'].fillna(0)

        # NEW: Log return volatility for more accurate volatility measurement
        df['log_volatility'] = df['log_returns'].rolling(window=20).std()
        df['log_volatility'] = df['log_volatility'].fillna(0)

        # Calculate Simple Moving Averages
        df['SMA20'] = df['4. close'].rolling(window=20).mean()
        df['SMA50'] = df['4. close'].rolling(window=50).mean()
        df['SMA100'] = df['4. close'].rolling(window=100).mean()
        df['SMA200'] = df['4. close'].rolling(window=200).mean()

        # Fill NaN values in SMAs with forward fill then backward fill
        for col in ['SMA20', 'SMA50', 'SMA100', 'SMA200']:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

        # Calculate Relative Strength Index (RSI)
        delta = df['4. close'].diff()
        delta = delta.fillna(0)

        # Handle division by zero and NaN values in RSI calculation
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        # Handle zero avg_loss
        rs = np.zeros_like(avg_gain)
        valid_indices = avg_loss != 0
        rs[valid_indices] = avg_gain[valid_indices] / avg_loss[valid_indices]

        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)  # Default to neutral RSI (50)

        # Calculate Bollinger Bands
        df['BB_middle'] = df['SMA20']
        df['BB_std'] = df['4. close'].rolling(window=20).std()
        df['BB_std'] = df['BB_std'].fillna(0)
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)

        # Calculate MACD
        df['EMA12'] = df['4. close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['4. close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Calculate trading volume changes
        if 'volume' in df.columns:
            df['volume_change'] = df['volume'].pct_change()
            df['volume_change'] = df['volume_change'].fillna(0)

        # MEAN REVERSION COMPONENTS

        # 1. Distance from SMA200 as mean reversion indicator
        df['dist_from_SMA200'] = (df['4. close'] / df['SMA200']) - 1

        # 2. Bollinger Band %B (0-1 scale where >1 is overbought, <0 is oversold)
        bb_range = df['BB_upper'] - df['BB_lower']
        df['BB_pctB'] = np.where(
            bb_range > 0,
            (df['4. close'] - df['BB_lower']) / bb_range,
            0.5
        )

        # 3. Price Rate of Change (historical returns over different periods)
        df['ROC_5'] = df['4. close'].pct_change(5)
        df['ROC_10'] = df['4. close'].pct_change(10)
        df['ROC_20'] = df['4. close'].pct_change(20)
        df['ROC_60'] = df['4. close'].pct_change(60)

        # 4. Overbought/Oversold indicator based on historical returns
        # Standardize recent returns relative to their own history
        returns_z_score = lambda x: (x - x.rolling(60).mean()) / x.rolling(60).std()
        df['returns_zscore_5'] = returns_z_score(df['ROC_5'])
        df['returns_zscore_20'] = returns_z_score(df['ROC_20'])

        # 5. Price acceleration (change in ROC) - detects momentum exhaustion
        df['ROC_accel'] = df['ROC_5'] - df['ROC_5'].shift(5)

        # 6. Historical volatility ratio (recent vs long-term)
        df['vol_ratio'] = df['volatility'] / df['volatility'].rolling(60).mean()

        # 7. Mean reversion potential based on distance from long-term trend
        # Using Z-score of price deviation from 200-day SMA
        mean_dist = df['dist_from_SMA200'].rolling(100).mean()
        std_dist = df['dist_from_SMA200'].rolling(100).std()
        df['mean_reversion_z'] = np.where(
            std_dist > 0,
            (df['dist_from_SMA200'] - mean_dist) / std_dist,
            0
        )

        # 8. RSI divergence (price making new highs but RSI isn't)
        df['price_high'] = df['4. close'].rolling(10).max() == df['4. close']
        df['rsi_high'] = df['RSI'].rolling(10).max() == df['RSI']
        # Potential negative divergence: price high but RSI not high
        df['rsi_divergence'] = np.where(df['price_high'] & ~df['rsi_high'], -1, 0)

        # 9. Volume-price relationship (high returns with low volume can signal exhaustion)
        if 'volume' in df.columns:
            df['vol_price_ratio'] = np.where(
                df['returns'] != 0,
                df['volume'] / (abs(df['returns']) * df['4. close']),
                0
            )
            df['vol_price_ratio_z'] = (df['vol_price_ratio'] - df['vol_price_ratio'].rolling(20).mean()) / df[
                'vol_price_ratio'].rolling(20).std()
        else:
            # Generate dummy columns if volume not available
            df['vol_price_ratio'] = 0
            df['vol_price_ratio_z'] = 0

        # 10. Stochastic Oscillator
        window = 14
        df['14-high'] = df['high'].rolling(window).max()
        df['14-low'] = df['low'].rolling(window).min()
        df['%K'] = (df['4. close'] - df['14-low']) * 100 / (df['14-high'] - df['14-low'])
        df['%D'] = df['%K'].rolling(3).mean()

        # 11. Advanced RSI Analysis
        # RSI slope (rate of change)
        df['RSI_slope'] = df['RSI'] - df['RSI'].shift(3)

        # RSI moving average crossovers
        df['RSI_MA5'] = df['RSI'].rolling(5).mean()
        df['RSI_MA14'] = df['RSI'].rolling(14).mean()

        # 12. Double Bollinger Bands (outer bands at 3 std dev)
        df['BB_upper_3'] = df['BB_middle'] + (df['BB_std'] * 3)
        df['BB_lower_3'] = df['BB_middle'] - (df['BB_std'] * 3)

        # 13. Volume Weighted MACD
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=14).mean()
            volume_ratio = np.where(df['volume_ma'] > 0, df['volume'] / df['volume_ma'], 1)
            df['vol_weighted_macd'] = df['MACD'] * volume_ratio
        else:
            df['volume_ma'] = 0
            df['vol_weighted_macd'] = df['MACD']

        # 14. Chaikin Money Flow (CMF)
        if 'volume' in df.columns:
            money_flow_multiplier = ((df['4. close'] - df['low']) - (df['high'] - df['4. close'])) / (
                    df['high'] - df['low'])
            money_flow_volume = money_flow_multiplier * df['volume']
            df['CMF'] = money_flow_volume.rolling(20).sum() / df['volume'].rolling(20).sum()
        else:
            df['CMF'] = 0

        # 15. Williams %R
        df['Williams_%R'] = -100 * (df['14-high'] - df['4. close']) / (df['14-high'] - df['14-low'])

        # 16. Advanced trend analysis
        df['trend_strength'] = np.abs(df['dist_from_SMA200'])
        df['price_vs_all_SMAs'] = np.where(
            (df['4. close'] > df['SMA20']) &
            (df['4. close'] > df['SMA50']) &
            (df['4. close'] > df['SMA100']) &
            (df['4. close'] > df['SMA200']),
            1, 0
        )

        # 17. SMA alignment (bullish/bearish alignment)
        df['sma_alignment'] = np.where(
            (df['SMA20'] > df['SMA50']) &
            (df['SMA50'] > df['SMA100']) &
            (df['SMA100'] > df['SMA200']),
            1,  # Bullish alignment
            np.where(
                (df['SMA20'] < df['SMA50']) &
                (df['SMA50'] < df['SMA100']) &
                (df['SMA100'] < df['SMA200']),
                -1,  # Bearish alignment
                0  # Mixed alignment
            )
        )

        # ======== NEW LOG RETURNS BASED MEAN REVERSION METRICS ========

        # 1. Log returns Z-score (more statistically valid than regular returns)
        log_returns_mean = df['log_returns'].rolling(100).mean()
        log_returns_std = df['log_returns'].rolling(100).std()
        df['log_returns_zscore'] = np.where(
            log_returns_std > 0,
            (df['log_returns'] - log_returns_mean) / log_returns_std,
            0
        )

        # 2. Log return mean reversion potential
        # Higher absolute values suggest stronger mean reversion potential
        # Sign indicates expected direction (negative means price likely to increase)
        df['log_mr_potential'] = -1 * df['log_returns_zscore']

        # 3. Log return autocorrelation - measures mean reversion strength
        # Uses 5-day lag as common mean-reversion period
        def autocorr(x, lag=5):
            if len(x.dropna()) <= lag:
                return 0
            x = x.dropna()
            return pd.Series(x).autocorr(lag=lag)

        df['log_autocorr_5'] = df['log_returns'].rolling(30).apply(
            lambda x: autocorr(x, lag=5),
            raw=False
        )

        # 4. Log volatility ratio (indicates regime changes)
        df['log_vol_ratio'] = df['log_volatility'] / df['log_volatility'].rolling(60).mean()

        # 5. Log return momentum vs mean reversion balance
        # This combines both momentum and mean reversion signals
        # Positive values suggest momentum dominates, negative suggest mean reversion dominates
        df['log_mom_vs_mr'] = df['log_returns'].rolling(10).mean() / df['log_volatility'] + df['log_autocorr_5']

        # 6. Log-based adaptive Bollinger Bands
        # More accurate for capturing true statistical extremes
        log_price = np.log(df['4. close'])
        log_ma20 = log_price.rolling(20).mean()
        log_std20 = log_price.rolling(20).std()
        df['log_bb_upper'] = np.exp(log_ma20 + 2 * log_std20)
        df['log_bb_lower'] = np.exp(log_ma20 - 2 * log_std20)
        df['log_bb_pctB'] = np.where(
            (df['log_bb_upper'] - df['log_bb_lower']) > 0,
            (df['4. close'] - df['log_bb_lower']) / (df['log_bb_upper'] - df['log_bb_lower']),
            0.5
        )

        # 7. Log return expected mean reversion magnitude
        # Estimates expected price change if fully reverted to mean
        df['log_expected_reversion'] = -1 * df['log_returns_zscore'] * df['log_volatility'] * np.sqrt(252)
        df['log_expected_reversion_pct'] = (np.exp(df['log_expected_reversion']) - 1) * 100

        # Fill NaN values in new indicators
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

        print(f"[DEBUG] Enhanced technical indicators with log returns calculated successfully. New shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Error calculating enhanced technical indicators: {e}")
        traceback.print_exc()
        return None


# Improved Hurst Exponent calculation using log returns
def calculate_hurst_exponent(df, max_lag=120, use_log_returns=True):
    """Calculate Hurst exponent to determine mean reversion vs trending behavior
       Now uses log returns for more accurate measurement"""
    try:
        # Use log returns for better statistical properties
        if use_log_returns and 'log_returns' in df.columns:
            returns = df['log_returns'].dropna().values
            print("[INFO] Using log returns for Hurst calculation")
        elif 'returns' in df.columns:
            returns = df['returns'].dropna().values
            print("[INFO] Using regular returns for Hurst calculation")
        else:
            # Fallback to calculating returns from price if neither is available
            price_col = 'close' if 'close' in df.columns else '4. close'
            prices = df[price_col].dropna().values
            returns = np.diff(np.log(prices))
            print("[INFO] Calculated returns from price data for Hurst calculation")

        if len(returns) < max_lag:
            print(f"[WARNING] Not enough returns for Hurst calculation: {len(returns)} < {max_lag}")
            max_lag = max(30, len(returns) // 4)  # Adjust max_lag if not enough data

        lags = range(2, max_lag)
        tau = []
        var = []

        for lag in lags:
            # Price (or return) difference
            pp = np.subtract(returns[lag:], returns[:-lag])
            # Variance
            variance = np.std(pp)
            var.append(variance)
            tau.append(lag)

        # Linear fit in log-log space to calculate Hurst exponent
        m = np.polyfit(np.log(tau), np.log(var), 1)
        hurst = m[0] / 2.0

        # Categorize by Hurst value
        if hurst < 0.4:
            regime = "Strong Mean Reversion"
        elif hurst < 0.45:
            regime = "Mean Reversion"
        elif hurst < 0.55:
            regime = "Random Walk"
        elif hurst < 0.65:
            regime = "Trending"
        else:
            regime = "Strong Trending"

        return {"hurst": hurst, "regime": regime}
    except Exception as e:
        print(f"[ERROR] Error calculating Hurst exponent: {e}")
        return {"hurst": 0.5, "regime": "Unknown"}


# Improved Mean Reversion Half-Life using log returns
def calculate_mean_reversion_half_life(data):
    """Estimate half-life of mean reversion using log returns with Ornstein-Uhlenbeck process"""
    try:
        # Check if we have log returns available, otherwise calculate them
        if 'log_returns' not in data.columns:
            log_returns = np.log(data['4. close'] / data['4. close'].shift(1)).dropna()
            print("[INFO] Calculating log returns for mean reversion half-life")
        else:
            log_returns = data['log_returns'].dropna()
            print("[INFO] Using existing log returns for mean reversion half-life")

        # Calculate deviation of log returns from their moving average
        ma = log_returns.rolling(window=50).mean()
        spread = log_returns - ma

        # Remove NaN values
        spread = spread.dropna()

        if len(spread) < 50:
            print("[WARNING] Not enough data for mean reversion half-life calculation")
            return {"half_life": 0, "mean_reversion_speed": "Unknown"}

        # Calculate autoregression coefficient
        # S_t+1 - S_t = a * S_t + e_t
        spread_lag = spread.shift(1).dropna()
        spread_current = spread.iloc[1:]

        # Match lengths
        spread_lag = spread_lag.iloc[:len(spread_current)]

        # Use regression to find the coefficient
        model = LinearRegression()
        model.fit(spread_lag.values.reshape(-1, 1), spread_current.values)
        beta = model.coef_[0]

        # Calculate half-life
        # The closer beta is to -1, the faster the mean reversion
        # If beta > 0, it's trending, not mean-reverting
        if -1 < beta < 0:
            half_life = -np.log(2) / np.log(1 + beta)
        else:
            # If beta is positive (momentum) or <= -1 (oscillatory), default to 0
            half_life = 0

        # Categorize strength
        if 0 < half_life <= 5:
            strength = "Very Fast"
        elif half_life <= 20:
            strength = "Fast"
        elif half_life <= 60:
            strength = "Medium"
        elif half_life <= 120:
            strength = "Slow"
        else:
            strength = "Very Slow or None"

        # Return beta for additional context
        return {
            "half_life": half_life,
            "mean_reversion_speed": strength,
            "beta": beta  # Added beta coefficient
        }
    except Exception as e:
        print(f"[ERROR] Error calculating mean reversion half-life: {e}")
        return {"half_life": 0, "mean_reversion_speed": "Unknown", "beta": 0}


# Volatility Regime Analysis with log-based improvements
def analyze_volatility_regimes(data, lookback=252):
    """Implements advanced volatility analysis with log returns for better accuracy"""
    try:
        # Ensure we have a DataFrame
        if not isinstance(data, pd.DataFrame):
            print("[WARNING] Input to analyze_volatility_regimes is not a DataFrame")
            if hasattr(data, '__array__'):
                data = pd.DataFrame(data)
            else:
                print("[ERROR] Cannot convert input to DataFrame, returning default")
                return {
                    'vol_term_structure': 0,
                    'vol_persistence': 0.5,
                    'vol_regime': "Stable"
                }

        # Handle empty DataFrame
        if data.empty:
            print("[WARNING] Empty DataFrame in analyze_volatility_regimes")
            return {
                'vol_term_structure': 0,
                'vol_persistence': 0.5,
                'vol_regime': "Stable"
            }

        # Use log returns if available for improved statistical properties
        if 'log_returns' in data.columns:
            returns = data['log_returns'].iloc[-lookback:]
            print("[INFO] Using log returns for volatility regime analysis")
        elif 'returns' in data.columns:
            returns = data['returns'].iloc[-lookback:]
            print("[INFO] Using regular returns for volatility regime analysis")
        else:
            # Fallback to calculating returns from price if neither is available
            price_col = 'close' if 'close' in data.columns else '4. close'
            if price_col not in data.columns:
                print(f"[WARNING] No price column found in data for volatility regime analysis")
                # Create synthetic returns for a fallback
                returns = pd.Series(np.random.normal(0, 0.01, min(lookback, len(data))),
                                    index=data.index[-min(lookback, len(data)):])
                print("[INFO] Created synthetic returns for volatility regime analysis")
            else:
                prices = data[price_col].iloc[-lookback - 1:].values
                log_returns = np.diff(np.log(prices))
                returns = pd.Series(log_returns, index=data.index[-len(log_returns):])
                print("[INFO] Calculated returns from price data for volatility regime analysis")

        # 1. Volatility term structure
        short_vol = returns.iloc[-20:].std() * np.sqrt(252)
        medium_vol = returns.iloc[-60:].std() * np.sqrt(252) if len(returns) >= 60 else short_vol
        long_vol = returns.iloc[-120:].std() * np.sqrt(252) if len(returns) >= 120 else medium_vol

        # Relative readings
        vol_term_structure = short_vol / long_vol
        vol_acceleration = (short_vol / medium_vol) / (medium_vol / long_vol)

        # 2. Parkinson volatility estimator (uses high-low range)
        if 'high' in data.columns and 'low' in data.columns:
            # Improved Parkinson estimator using log prices
            high_low_ratio = np.log(data['high'] / data['low'])
            parker_vol = np.sqrt(1 / (4 * np.log(2)) * high_low_ratio.iloc[-20:].pow(2).mean() * 252)
        else:
            parker_vol = None

        # 3. GARCH-like volatility persistence estimation
        try:
            # Simple AR(1) model to estimate volatility persistence
            squared_returns = returns.pow(2).dropna()
            if len(squared_returns) > 22:  # At least a month of data
                sq_ret_lag = squared_returns.shift(1).dropna()
                sq_ret = squared_returns.iloc[1:]

                # Match lengths
                sq_ret_lag = sq_ret_lag.iloc[:len(sq_ret)]

                if len(sq_ret) > 10:  # Need sufficient data
                    # Fit AR(1) model to squared returns
                    vol_model = LinearRegression()
                    vol_model.fit(sq_ret_lag.values.reshape(-1, 1), sq_ret.values)
                    vol_persistence = vol_model.coef_[0]  # How much volatility persists
                else:
                    vol_persistence = 0.8  # Default value
            else:
                vol_persistence = 0.8  # Default value
        except:
            vol_persistence = 0.8  # Default if calculation fails

        # Volatility regime detection
        if vol_term_structure > 1.3:
            vol_regime = "Rising"
        elif vol_term_structure < 0.7:
            vol_regime = "Falling"
        else:
            vol_regime = "Stable"

        return {
            "vol_term_structure": vol_term_structure,
            "vol_acceleration": vol_acceleration,
            "parkinson_vol": parker_vol,
            "vol_regime": vol_regime,
            "vol_persistence": vol_persistence,  # New metric
            "short_vol": short_vol,
            "medium_vol": medium_vol,
            "long_vol": long_vol
        }
    except Exception as e:
        print(f"[ERROR] Error analyzing volatility regimes: {e}")
        # Fallback in case of calculation issues
        return {
            "vol_regime": "Unknown",
            "vol_term_structure": 1.0,
            "vol_persistence": 0.8
        }


# Market Regime Detection with log returns
def detect_market_regime(data, n_regimes=3):
    """Detect market regimes using Hidden Markov Model on log returns for improved results"""
    try:
        # Extract features for regime detection
        try:
            # Prepare the returns data
            if 'log_returns' in data.columns:
                returns = data['log_returns'].fillna(0).values.reshape(-1, 1)
                print("[INFO] Using log returns for market regime detection")
            elif 'returns' in data.columns:
                returns = data['returns'].fillna(0).values.reshape(-1, 1)
                print("[INFO] Using regular returns for market regime detection")
            else:
                # Calculate returns from close prices
                if '4. close' in data.columns:
                    prices = data['4. close'].values
                    returns = np.diff(np.log(prices)).reshape(-1, 1)
                    returns = np.concatenate([np.zeros((1, 1)), returns])  # Add a zero for the first row
                    print("[INFO] Calculated log returns from close prices for market regime detection")
                else:
                    print("[ERROR] No return or price data found for market regime detection")
                    raise ValueError("No return or price data available")

            # Fit HMM with fewer iterations for performance
            model = hmm.GaussianHMM(n_components=n_regimes, n_iter=100, random_state=42)
            model.fit(returns)

            # Predict regime
            hidden_states = model.predict(returns)

            # Map states to meaningful regimes
            states_volatility = {}
            for state in range(n_regimes):
                state_returns = returns[hidden_states == state]
                states_volatility[state] = np.std(state_returns)

            # Sort states by volatility
            sorted_states = sorted(states_volatility.items(), key=lambda x: x[1])
            regime_map = {}
            regime_map[sorted_states[0][0]] = "Low Volatility"
            regime_map[sorted_states[-1][0]] = "High Volatility"

            if n_regimes > 2:
                for i in range(1, n_regimes - 1):
                    regime_map[sorted_states[i][0]] = f"Medium Volatility {i}"

            # Get current regime
            current_regime = regime_map[hidden_states[-1]]

            # Calculate regime stability (how long we've been in this regime)
            regime_duration = 1
            for i in range(2, min(100, len(hidden_states))):
                if hidden_states[-i] == hidden_states[-1]:
                    regime_duration += 1
                else:
                    break

            return {
                "current_regime": current_regime,
                "regime_duration": regime_duration,
                "regime_volatility": states_volatility[hidden_states[-1]]
            }
        except Exception as e:
            print(f"[ERROR] Error in market regime detection: {e}")
            raise ValueError(f"Market regime detection failed: {e}")
    except Exception as e:
        print(f"[ERROR] Error in market regime detection outer handler: {e}")
        return {
            "current_regime": "Unknown",
            "regime_duration": 0,
            "regime_volatility": 0
        }


# Risk-Adjusted Metrics with log return improvements
# 3. Replace the old Risk-Adjusted Metrics function with our enhanced version
def calculate_risk_adjusted_metrics(df, sigma):
    """Calculate risk-adjusted metrics using log returns and enhanced drawdown analysis"""
    try:
        # Use log returns if available for better statistical properties
        if 'log_returns' in df.columns:
            returns = df['log_returns'].dropna()
            print("[INFO] Using log returns for risk-adjusted metrics")
        elif 'returns' in df.columns:
            returns = df['returns'].dropna()
            print("[INFO] Using regular returns for risk-adjusted metrics")
        else:
            # Fallback to calculating returns from price
            price_col = 'close' if 'close' in df.columns else '4. close'
            prices = df[price_col].dropna().values
            returns = pd.Series(np.diff(np.log(prices)))
            print("[INFO] Calculated returns from price data for risk-adjusted metrics")

        # Run enhanced drawdown analysis
        drawdown_analysis = analyze_drawdowns(df, window_size=180)  # Matching LSTM window for consistency
        
        # Get adaptive risk adjustment factor from drawdown analysis
        risk_adjustment = drawdown_analysis.get('risk_adjustment', 0.8)
        max_drawdown = drawdown_analysis.get('max_drawdown', 0)

        # Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
        alpha = 0.05  # 95% confidence level
        var_95 = np.percentile(returns, alpha * 100)
        cvar_95 = returns[returns <= var_95].mean() if sum(returns <= var_95) > 0 else var_95

        # Calculate Kelly Criterion
        # For log returns, adjust the win/loss calculation
        if 'log_returns' in df.columns:
            # Convert to arithmetic returns for Kelly calculation
            arith_returns = np.exp(returns) - 1
            win_rate = len(arith_returns[arith_returns > 0]) / len(arith_returns)
            avg_win = arith_returns[arith_returns > 0].mean() if len(arith_returns[arith_returns > 0]) > 0 else 0
            avg_loss = abs(arith_returns[arith_returns < 0].mean()) if len(arith_returns[arith_returns < 0]) > 0 else 0
        else:
            win_rate = len(returns[returns > 0]) / len(returns)
            avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0

        # Avoid division by zero
        if avg_loss > 0:
            kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        else:
            kelly = win_rate

        # Scale kelly to avoid extreme values
        kelly = max(-1, min(1, kelly))

        # Calculate Sharpe Ratio (annualized) using log returns for better properties
        if 'log_returns' in df.columns:
            # For log returns, we need to annualize differently
            ret_mean = returns.mean() * 252
            ret_std = returns.std() * np.sqrt(252)
        else:
            ret_mean = returns.mean() * 252
            ret_std = returns.std() * np.sqrt(252)

        sharpe = ret_mean / ret_std if ret_std > 0 else 0

        # Calculate Sortino Ratio (focusing on downside risk)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else ret_std
        sortino = ret_mean / downside_std if downside_std > 0 else 0

        # Scale sigma based on risk metrics and drawdown analysis
        risk_adjusted_sigma = sigma * risk_adjustment

        # Reduce sigma for negative kelly values on top of drawdown adjustment
        if kelly < 0:
            risk_adjusted_sigma *= (1 + kelly)  # Reduce by up to 100% for kelly = -1

        # Adjust based on Sharpe ratio
        if sharpe > 1.5:  # Excellent risk-adjusted returns
            risk_adjusted_sigma = min(1.0, risk_adjusted_sigma * 1.2)  # Boost by up to 20%
        elif sharpe < 0:  # Poor risk-adjusted returns
            risk_adjusted_sigma *= 0.8  # Reduce by 20%

        # Ensure sigma is within bounds
        risk_adjusted_sigma = max(0.01, min(1.0, risk_adjusted_sigma))

        return {
            "max_drawdown": max_drawdown,
            "cvar_95": cvar_95,
            "kelly": kelly,
            "sharpe": sharpe,
            "sortino": sortino,
            "risk_adjusted_sigma": risk_adjusted_sigma,
            "drawdown_analysis": drawdown_analysis
        }
    except Exception as e:
        print(f"[ERROR] Error calculating risk-adjusted metrics: {e}")
        traceback.print_exc()
        return {
            "max_drawdown": max_drawdown if 'max_drawdown' in locals() else 0,
            "cvar_95": 0,
            "kelly": 0,
            "sharpe": 0,
            "sortino": 0,
            "risk_adjusted_sigma": sigma * 0.8  # Default conservative adjustment
        }

# PCA function to reduce dimensionality of features
def apply_pca(features_df):
    try:
        # Debug info about input
        print(f"[DEBUG] PCA input shape: {features_df.shape}")

        # Check if we have enough data
        if features_df.shape[0] < 10 or features_df.shape[1] < 5:
            print(f"[WARNING] Not enough data for PCA analysis: {features_df.shape}")
            return None, None

        # Select numerical columns that aren't NaN
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude columns that are mostly NaN
        valid_cols = []
        for col in numeric_cols:
            if features_df[col].isna().sum() < len(
                    features_df) * 0.3:  # At least 70% of values are not NaN
                valid_cols.append(col)

        if len(valid_cols) < 5:
            print(f"[WARNING] Not enough valid columns for PCA: {len(valid_cols)}")
            return None, None

        numeric_df = features_df[valid_cols].copy()

        # Fill remaining NaN values with column means
        for col in numeric_df.columns:
            if numeric_df[col].isna().any():
                numeric_df[col] = numeric_df[col].fillna(numeric_df[col].mean())

        print(f"[DEBUG] PCA numeric data shape after cleaning: {numeric_df.shape}")

        # Check for remaining NaN values
        if numeric_df.isna().sum().sum() > 0:
            print(f"[WARNING] NaN values still present after cleaning: {numeric_df.isna().sum().sum()}")
            # Replace remaining NaNs with 0
            numeric_df = numeric_df.fillna(0)

        # Standardize the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # Apply PCA
        n_components = min(10, min(scaled_data.shape) - 1)  # Increased from 5
        pca = PCA(n_components=n_components)
        pca_results = pca.fit_transform(scaled_data)

        # Create a DataFrame with PCA results
        pca_df = pd.DataFrame(
            pca_results,
            columns=[f'PC{i + 1}' for i in range(pca_results.shape[1])],
            index=features_df.index
        )

        # Calculate explained variance for each component
        explained_variance = pca.explained_variance_ratio_

        print(f"[INFO] PCA explained variance: {explained_variance}")
        return pca_df, explained_variance
    except Exception as e:
        print(f"[ERROR] PCA failed: {e}")
        traceback.print_exc()
        return None, None


# 1. First modification: Change LSTM time steps to 180 days
def prepare_lstm_data(data, time_steps=180):  # Changed from 60 to 180
    try:
        # Check if we have enough data
        if len(data) < time_steps + 10:
            print(f"[WARNING] Not enough data for LSTM: {len(data)} < {time_steps + 10}")
            return None, None, None

        # Rest of the function remains the same
        # Use multiple features including log returns
        features = []

        # Always include closing price
        features.append(data['4. close'].values)

        # Include log returns if available (preferred)
        if 'log_returns' in data.columns:
            features.append(data['log_returns'].values)
            print("[INFO] Using log returns for LSTM features")
        # Otherwise use regular returns
        elif 'returns' in data.columns:
            features.append(data['returns'].values)
            print("[INFO] Using regular returns for LSTM features (log returns not available)")

        # Include volume if available with appropriate scaling
        if 'volume' in data.columns:
            # Log transform volume to reduce scale differences
            log_volume = np.log1p(data['volume'].values)
            features.append(log_volume)

        # Include log volatility if available (preferred)
        if 'log_volatility' in data.columns:
            features.append(data['log_volatility'].values)
            print("[INFO] Using log volatility for LSTM features")
        # Otherwise use regular volatility
        elif 'volatility' in data.columns:
            features.append(data['volatility'].values)
            print("[INFO] Using regular volatility for LSTM features (log volatility not available)")

        # Include RSI if available
        if 'RSI' in data.columns:
            # Normalize RSI to 0-1 scale
            normalized_rsi = data['RSI'].values / 100
            features.append(normalized_rsi)

        # Include MACD if available
        if 'MACD' in data.columns:
            # Normalize MACD using tanh for -1 to 1 range
            normalized_macd = np.tanh(data['MACD'].values / 5)
            features.append(normalized_macd)

        # Include log-based mean reversion indicators if available
        if 'log_returns_zscore' in data.columns:
            # Normalize with tanh to -1 to 1 range
            log_returns_z = np.tanh(data['log_returns_zscore'].values)
            features.append(log_returns_z)
            print("[INFO] Adding log returns z-score to LSTM features")

        if 'log_mr_potential' in data.columns:
            # Already normalized
            features.append(data['log_mr_potential'].values)
            print("[INFO] Adding log mean reversion potential to LSTM features")

        if 'log_expected_reversion_pct' in data.columns:
            # Normalize with tanh
            log_exp_rev = np.tanh(data['log_expected_reversion_pct'].values / 10)
            features.append(log_exp_rev)
            print("[INFO] Adding log expected reversion to LSTM features")

        # Include regular mean reversion indicators as fallback
        if 'BB_pctB' in data.columns and 'log_bb_pctB' not in data.columns:
            features.append(data['BB_pctB'].values)

        if 'dist_from_SMA200' in data.columns:
            # Use tanh to normalize to -1 to 1 range
            normalized_dist = np.tanh(data['dist_from_SMA200'].values * 5)
            features.append(normalized_dist)

        # Include Williams %R if available
        if 'Williams_%R' in data.columns:
            # Normalize from -100-0 to 0-1
            normalized_williams = (data['Williams_%R'].values + 100) / 100
            features.append(normalized_williams)

        # Include CMF if available
        if 'CMF' in data.columns:
            # Already in -1 to 1 range
            features.append(data['CMF'].values)

        # Stack features
        feature_array = np.column_stack(features)

        # Check for NaN values across all features
        if np.isnan(feature_array).any():
            print(f"[WARNING] NaN values in features, filling with forward fill")
            # Convert to DataFrame for easier handling of NaNs
            temp_df = pd.DataFrame(feature_array)
            # Fill NaN values
            temp_df = temp_df.fillna(method='ffill').fillna(method='bfill')
            feature_array = temp_df.values

        # Normalize the data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_array)

        # Create sequences with all features
        X, y = [], []
        # Target is still the closing price (first feature)
        for i in range(len(scaled_features) - time_steps):
            X.append(scaled_features[i:i + time_steps])
            # For prediction target, use only the closing price column (index 0)
            y.append(scaled_features[i + time_steps, 0:1])

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Check shapes
        print(f"[DEBUG] Enhanced LSTM data shapes with log returns: X={X.shape}, y={y.shape}")

        return X, y, scaler
    except Exception as e:
        print(f"[ERROR] Error preparing enhanced LSTM data: {e}")
        traceback.print_exc()
        # Fallback to simpler preparation if enhanced fails
        try:
            print(f"[WARNING] Falling back to simple price-only LSTM preparation")
            # Get closing prices only
            prices = data['4. close'].values

            # Handle NaN values
            if np.isnan(prices).any():
                prices = pd.Series(prices).fillna(method='ffill').fillna(method='bfill').values

            # Reshape and scale
            prices_2d = prices.reshape(-1, 1)
            scaler = StandardScaler()
            scaled_prices = scaler.fit_transform(prices_2d)

            # Create sequences
            X, y = [], []
            for i in range(len(scaled_prices) - time_steps):
                X.append(scaled_prices[i:i + time_steps])
                y.append(scaled_prices[i + time_steps])

            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)

            print(f"[DEBUG] Fallback LSTM data shapes: X={X.shape}, y={y.shape}")
            return X, y, scaler

        except Exception as e2:
            print(f"[ERROR] Fallback LSTM data preparation also failed: {e2}")
            return None, None, None

# Enhanced LSTM model for volatility prediction - maximized for M1 iMac
def build_lstm_model(input_shape):
    try:
        # Highly sophisticated architecture for maximum prediction accuracy
        inputs = Input(shape=input_shape)

        # First LSTM layer with more units
        x = LSTM(128, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Second LSTM layer
        x = LSTM(128, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Third LSTM layer
        x = LSTM(64, return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Dense layers for feature extraction
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Final dense layer before output
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)

        # Output layer
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)

        # Use Adam optimizer with custom learning rate
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mse")
        return model
    except Exception as e:
        print(f"[ERROR] Error building enhanced LSTM model: {e}")
        traceback.print_exc()

        # Fallback to simpler model if complex one fails
        try:
            inputs = Input(shape=input_shape)
            x = LSTM(64, return_sequences=True)(inputs)
            x = Dropout(0.2)(x)
            x = LSTM(64, return_sequences=False)(x)
            x = Dense(32, activation='relu')(x)
            outputs = Dense(1)(x)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer="adam", loss="mse")
            return model
        except Exception as e2:
            print(f"[ERROR] Fallback LSTM model also failed: {e2}")

            # Very simple fallback
            try:
                inputs = Input(shape=input_shape)
                x = LSTM(32, return_sequences=False)(inputs)
                outputs = Dense(1)(x)
                model = Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer="adam", loss="mse")
                return model
            except Exception as e3:
                print(f"[ERROR] All LSTM model attempts failed: {e3}")
                return None


# Enhanced LSTM model training and prediction with extended processing time
# 4. Update prediction with LSTM to use 180-day window
def predict_with_lstm(data):
    try:
        # Set a maximum execution time - significantly increased for thorough training
        max_execution_time = 240  # 4 minutes max (increased from 2 minutes)
        start_time = time.time()

        # Require less data to attempt prediction
        if len(data) < 180:  # Update minimum data requirement for 180-day window
            print("[WARNING] Not enough data for LSTM model with 180-day window")
            return 0

        # Use updated window size
        time_steps = 180  # Increased from 60 to 180 for better prediction accuracy

        # Prepare data with enhanced features including log returns
        X, y, scaler = prepare_lstm_data(data, time_steps=time_steps)
        if X is None or y is None or scaler is None:
            print("[WARNING] Failed to prepare LSTM data")
            return 0

        # More lenient on required data size
        if len(X) < 8:
            print(f"[WARNING] Not enough data after preparation: {len(X)}")
            return 0

        # Build enhanced model - adjust input shape for larger time window
        model = build_lstm_model((X.shape[1], X.shape[2]))
        if model is None:
            print("[WARNING] Failed to build LSTM model")
            return 0

        # Use more training data for better learning
        max_samples = 1000
        if len(X) > max_samples:
            # Use evenly spaced samples to get good representation
            indices = np.linspace(0, len(X) - 1, max_samples, dtype=int)
            X_train = X[indices]
            y_train = y[indices]
        else:
            X_train = X
            y_train = y

        # Rest of the function remains the same
        # Use try/except for model training
        try:
            # Check if we're still within time limit
            if time.time() - start_time > max_execution_time:
                print("[WARNING] LSTM execution time limit reached before training")
                # Use a better fallback prediction based on recent volatility
                if 'log_volatility' in data.columns:
                    return data['log_volatility'].iloc[-15:].mean() / data['log_volatility'].iloc[-45:].mean()
                else:
                    return data['volatility'].iloc[-15:].mean() / data['volatility'].iloc[-45:].mean()

            # Train model with more epochs and better callbacks
            early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0)
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.0001)

            # Set parameters for extensive training
            model.fit(
                X_train, y_train,
                epochs=30,
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0,
                shuffle=True
            )

            # Extra training round with lower learning rate for fine-tuning
            if time.time() - start_time < max_execution_time * 0.6:
                # Reduce learning rate for fine-tuning
                for layer in model.layers:
                    if hasattr(layer, 'optimizer'):
                        layer.optimizer.lr = layer.optimizer.lr * 0.3

                model.fit(
                    X_train, y_train,
                    epochs=20,
                    batch_size=32,
                    verbose=0,
                    shuffle=True
                )

            # Final fine-tuning with small batch size if time permits
            if time.time() - start_time < max_execution_time * 0.8:
                model.fit(
                    X_train, y_train,
                    epochs=10,
                    batch_size=16,  # Smaller batch size for final tuning
                    verbose=0,
                    shuffle=True
                )

        except Exception as e:
            print(f"[ERROR] LSTM model training failed: {e}")
            return 0

        # Make prediction for future volatility
        try:
            # Check time again
            if time.time() - start_time > max_execution_time:
                print("[WARNING] LSTM execution time limit reached before prediction")
                return 0.5  # Return a neutral value

            # Use ensemble of predictions from the last few sequences for better stability
            num_pred_samples = min(10, len(X))
            predictions = []

            for i in range(num_pred_samples):
                seq_idx = len(X) - i - 1
                if seq_idx >= 0:  # Check if index is valid
                    sequence = X[seq_idx].reshape(1, X.shape[1], X.shape[2])
                    pred = model.predict(sequence, verbose=0)[0][0]
                    predictions.append(pred)

            if not predictions:
                return 0.5  # Default if no valid predictions

            # Weight more recent predictions higher
            weights = np.linspace(1.0, 0.5, len(predictions))
            weights = weights / np.sum(weights)  # Normalize

            avg_prediction = np.sum(np.array(predictions) * weights)

            # Get weighted average of recent actual values
            last_actuals = y[-num_pred_samples:].flatten()
            last_actual_weights = np.linspace(1.0, 0.5, len(last_actuals))
            last_actual_weights = last_actual_weights / np.sum(last_actual_weights)
            last_actual = np.sum(last_actuals * last_actual_weights)

            # Avoid division by zero
            if abs(last_actual) < 1e-6:
                predicted_volatility_change = abs(avg_prediction)
            else:
                predicted_volatility_change = abs((avg_prediction - last_actual) / last_actual)

            print(f"[DEBUG] LSTM prediction with 180-day window: {predicted_volatility_change}")

            # Return a more nuanced measure capped at 1.0
            return min(1.0, max(0.1, predicted_volatility_change))

        except Exception as e:
            print(f"[ERROR] LSTM prediction failed: {e}")
            return 0
    except Exception as e:
        print(f"[ERROR] Error in LSTM prediction: {e}")
        traceback.print_exc()
        return 0

# Enhanced DQN Agent implementation for more accurate predictions
class DQNAgent:
    def __init__(self, action_size=3):
        """
        Initialize a DQN agent with dynamic model building

        Parameters:
        -----------
        action_size: int
            Number of possible actions (default: 3 for Buy, Hold, Sell)
        """
        # Core parameters
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.98  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.03  # Minimum exploration probability
        self.epsilon_decay = 0.97  # Exploration decay rate

        # Models will be built dynamically
        self.model = None
        self.target_model = None
        self.state_size = None

        # Training parameters
        self.target_update_counter = 0
        self.target_update_freq = 5
        self.max_training_time = 120  # 2 minutes maximum for training
        self.batch_history = []

        # Track start time for timeout purposes
        self.training_start_time = None

        print("[INFO] DQN agent initialized with dynamic model building")

    def _build_model(self, input_shape):
        """
        Build model based on actual input shape

        Parameters:
        -----------
        input_shape: int
            Number of input features

        Returns:
        --------
        keras.Model or None
            The compiled model, or None if building fails
        """
        self.state_size = input_shape
        print(f"[INFO] Building DQN model with input shape: {input_shape}")

        try:
            # Create model with proper input shape
            model = Sequential([
                Dense(128, input_dim=input_shape, activation='relu'),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(self.action_size, activation='linear')
            ])

            # Use a simple optimizer for better compatibility
            model.compile(optimizer='adam', loss='mse')

            # Report model size
            print(f"[INFO] Built DQN model with {model.count_params():,} parameters")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to build DQN model: {e}")
            traceback.print_exc()
            return None

    def ensure_model_built(self, state):
        """
        Make sure model is built with correct shape before use

        Parameters:
        -----------
        state: numpy.ndarray
            The state input to check dimensions against
        """
        try:
            # Get the shape from the state
            if len(state.shape) > 1:
                # Batch of states (e.g., [batch_size, features])
                input_shape = state.shape[1]
            else:
                # Single state (e.g., [features])
                input_shape = state.shape[0]

            if self.model is None:
                # Initial model build
                self.model = self._build_model(input_shape)
                self.target_model = self._build_model(input_shape)
                if self.model is not None and self.target_model is not None:
                    self.update_target_model()  # Sync weights initially
            elif input_shape != self.state_size:
                # Input shape changed - rebuild model
                print(f"[WARNING] Input shape changed from {self.state_size} to {input_shape}, rebuilding model")
                self.model = self._build_model(input_shape)
                self.target_model = self._build_model(input_shape)
                if self.model is not None and self.target_model is not None:
                    self.update_target_model()
        except Exception as e:
            print(f"[ERROR] Error ensuring model is built: {e}")
            traceback.print_exc()

    def act(self, state):
        """
        Select an action based on the current state

        Parameters:
        -----------
        state: numpy.ndarray
            Current state observation

        Returns:
        --------
        int
            Selected action index
        """
        try:
            # Reshape state if necessary
            if len(state.shape) == 1:
                state = state.reshape(1, -1)

            # Make sure model is built before using it
            self.ensure_model_built(state)

            if self.model is None:
                # Model building failed, return random action
                return random.randrange(self.action_size)

            # Epsilon-greedy action selection
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)

            # Use multiple predictions with noise for more stable selection
            try:
                # Get action values
                act_values = self.model.predict(state, verbose=0)
                # Add small noise for exploration
                act_values += np.random.normal(0, 0.05, size=act_values.shape)
                return np.argmax(act_values[0])
            except Exception as e:
                print(f"[WARNING] Error predicting action: {e}")
                return random.randrange(self.action_size)

        except Exception as e:
            print(f"[ERROR] Error in DQN act method: {e}")
            traceback.print_exc()
            return random.randrange(self.action_size)

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory

        Parameters:
        -----------
        state: numpy.ndarray
            Current state
        action: int
            Action taken
        reward: float
            Reward received
        next_state: numpy.ndarray
            Next state observed
        done: bool
            Whether episode is done
        """
        # Handle state shaping
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        if len(next_state.shape) == 1:
            next_state = next_state.reshape(1, -1)

        # Only add to memory if not full
        if len(self.memory) < self.memory.maxlen:
            # Convert to numpy arrays to ensure consistency
            state = np.array(state, dtype=np.float32)
            next_state = np.array(next_state, dtype=np.float32)

            # Store experience tuple
            self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        """Update target model with weights from the main model"""
        if self.model is not None and self.target_model is not None:
            try:
                self.target_model.set_weights(self.model.get_weights())
                print("[DEBUG] DQN target model updated")
            except Exception as e:
                print(f"[ERROR] Failed to update target model: {e}")

    def replay(self, batch_size):
        """
        Train the model using experience replay

        Parameters:
        -----------
        batch_size: int
            Number of experiences to sample for training
        """
        if len(self.memory) < batch_size or self.model is None:
            return

        # Add timeout mechanism
        start_time = time.time()
        self.training_start_time = self.training_start_time or start_time

        try:
            # Check for timeout
            if time.time() - self.training_start_time > self.max_training_time:
                print("[WARNING] DQN training timeout reached")
                return

            # Get random sample of experiences
            minibatch = random.sample(self.memory, batch_size)

            # Ensure all states in the batch have the same shape
            state_shapes = set(tuple(x[0].shape) for x in minibatch)
            next_state_shapes = set(tuple(x[3].shape) for x in minibatch)

            if len(state_shapes) > 1 or len(next_state_shapes) > 1:
                print(f"[WARNING] Mixed state shapes in batch: {state_shapes}, {next_state_shapes}, skipping training")
                return

            # Process batch data
            try:
                states = np.vstack([x[0] for x in minibatch])
                actions = np.array([x[1] for x in minibatch])
                rewards = np.array([x[2] for x in minibatch], dtype=np.float32)
                next_states = np.vstack([x[3] for x in minibatch])
                dones = np.array([x[4] for x in minibatch], dtype=np.bool_)
            except Exception as e:
                print(f"[ERROR] Error stacking batch data: {e}")
                return

            # Make sure model matches input shape
            self.ensure_model_built(states)

            if self.model is None or self.target_model is None:
                return

            # Get current Q values and target Q values
            try:
                # Current Q values
                targets = self.model.predict(states, verbose=0)

                # Get next Q values from target model for stability
                next_q_values = self.target_model.predict(next_states, verbose=0)

                # Update Q values using the Bellman equation
                for i in range(len(minibatch)):
                    if dones[i]:
                        targets[i, actions[i]] = rewards[i]
                    else:
                        # Add small noise to next Q-values for exploration
                        next_qs = next_q_values[i] + np.random.normal(0, 0.01, size=next_q_values[i].shape)
                        targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_qs)

                # Train the model
                history = self.model.fit(
                    states,
                    targets,
                    epochs=3,
                    batch_size=min(32, len(minibatch)),  # Smaller batches for stability
                    verbose=0
                )

                # Track training progress
                self.batch_history.append(history.history['loss'][-1])

                # Update epsilon with a more gradual decay
                if self.epsilon > self.epsilon_min:
                    # Adaptive decay based on memory size
                    decay_rate = self.epsilon_decay + (0.01 * min(1.0, len(self.memory) / 5000))
                    self.epsilon *= decay_rate
                    self.epsilon = max(self.epsilon, self.epsilon_min)  # Ensure we don't go below min

                # Update target network periodically
                self.target_update_counter += 1
                if self.target_update_counter >= self.target_update_freq:
                    self.update_target_model()
                    self.target_update_counter = 0

                # Report training progress
                if self.batch_history:
                    recent_losses = self.batch_history[-min(10, len(self.batch_history)):]
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    print(f"[DEBUG] DQN training - avg loss: {avg_loss:.5f}, epsilon: {self.epsilon:.3f}")

            except Exception as e:
                print(f"[ERROR] Error during model training: {e}")
                traceback.print_exc()

        except Exception as e:
            print(f"[ERROR] Error in DQN replay: {e}")
            traceback.print_exc()

    def get_recommendation(self, state):
        """
        Get recommendation score (0-1) based on current state

        Parameters:
        -----------
        state: numpy.ndarray
            Current state to evaluate

        Returns:
        --------
        float
            Recommendation score between 0 and 1
        """
        try:
            # Reshape state if necessary
            if len(state.shape) == 1:
                state = state.reshape(1, -1)

            # Ensure model is built
            self.ensure_model_built(state)

            if self.model is None:
                return 0.5  # Neutral recommendation if model failed

            # Get action values
            action_values = self.model.predict(state, verbose=0)[0]

            # Convert to probabilities using softmax
            exp_values = np.exp(action_values - np.max(action_values))
            probs = exp_values / np.sum(exp_values)

            # Calculate recommendation score (0=Sell, 0.5=Hold, 1=Buy)
            # Assuming 3 actions where index 0=Sell, 1=Hold, 2=Buy
            recommendation = 0.0 * probs[0] + 0.5 * probs[1] + 1.0 * probs[2]

            # Ensure score is between 0 and 1
            recommendation = max(0.0, min(1.0, recommendation))

            print(f"[INFO] DQN recommendation score: {recommendation:.3f} (action probs: {probs})")
            return recommendation

        except Exception as e:
            print(f"[ERROR] Error generating DQN recommendation: {e}")
            traceback.print_exc()
            return 0.5  # Neutral score on failure


# Enhanced DQN recommendation with log returns features
def get_dqn_recommendation(data):
    """Enhanced DQN recommendation with dynamic model building"""
    try:
        # More lenient on required data
        if len(data) < 40:
            print("[WARNING] Not enough data for DQN")
            return 0.5  # Neutral score

        # Set timeout for the entire function
        function_start_time = time.time()
        max_function_time = 2400  # 4 minutes

        # Prepare state features with more historical context
        lookback = 20  # Further increased from 10 for better historical context

        # Extract features for the state - keep your existing feature extraction code
        features = []

        # Basic indicators - prefer log returns if available
        if 'log_returns' in data.columns:
            features.append(data['log_returns'].values[-lookback:])
            print("[INFO] Using log returns for DQN features")
        elif 'returns' in data.columns:
            features.append(data['returns'].values[-lookback:])

        # Prefer log volatility if available
        if 'log_volatility' in data.columns:
            features.append(data['log_volatility'].values[-lookback:])
            print("[INFO] Using log volatility for DQN features")
        elif 'volatility' in data.columns:
            features.append(data['volatility'].values[-lookback:])

        # Add the rest of your features as in your original code...
        # Technical indicators
        if 'RSI' in data.columns:
            rsi = data['RSI'].values[-lookback:] / 100  # Normalize to 0-1
            features.append(rsi)
        if 'MACD' in data.columns:
            macd = np.tanh(data['MACD'].values[-lookback:] / 5)
            features.append(macd)
        # ... other features

        # Stack all features into the state
        features = [np.nan_to_num(f, nan=0.0) for f in features]  # Handle NaNs

        if len(features) > 0:
            current_state = np.concatenate(features)
        else:
            # Fallback if no features
            print("[WARNING] No features available for DQN, using default state")
            current_state = np.zeros(lookback * 3)  # Some reasonable default size

        # Initialize agent with only action_size
        # No need to specify state_size as it will be determined dynamically
        agent = DQNAgent(action_size=3)

        # Use more training data for better learning
        max_train_points = min(500, len(data) - (lookback + 1))
        step_size = max(1, (len(data) - (lookback + 1)) // 500)

        # First pass: collect experiences without training to populate memory
        print("[DEBUG] DQN collecting initial experiences with log returns...")
        experiences_collected = 0

        for i in range(0, max_train_points * step_size, step_size):
            # Check timeout
            if time.time() - function_start_time > max_function_time * 0.25:
                print(f"[WARNING] DQN experience collection timeout after {experiences_collected} experiences")
                break

            # Get index with bounds checking
            idx = min(i, len(data) - (lookback + 1))
            next_idx = min(idx + 1, len(data) - lookback - 1)

            # Extract features for this time point - Similar to what you did before
            try:
                # Your feature extraction code for historical data points
                past_features = []

                # Same feature extraction as above but for past data
                if 'log_returns' in data.columns:
                    values = data['log_returns'].values[idx:idx + lookback]
                    past_features.append(np.nan_to_num(values, nan=0.0))
                    print("[INFO] Using log returns for DQN reward calculation")
                elif 'returns' in data.columns:
                    values = data['returns'].values[idx:idx + lookback]
                    past_features.append(np.nan_to_num(values, nan=0.0))

                # Add other features too...

                # Create state vector and next state vector
                if len(past_features) > 0:
                    state_vector = np.concatenate(past_features)
                    next_state_vector = state_vector.copy()  # Simplified next state
                else:
                    state_vector = np.zeros_like(current_state)
                    next_state_vector = np.zeros_like(current_state)

                # Calculate reward - keep your existing reward logic
                try:
                    # Calculate reward based on future returns
                    if 'log_returns' in data.columns and next_idx + lookback < len(data):
                        price_return = data['log_returns'].values[next_idx + lookback - 1]
                    elif 'returns' in data.columns and next_idx + lookback < len(data):
                        price_return = data['returns'].values[next_idx + lookback - 1]
                    else:
                        price_return = 0

                    # Get current action for this state
                    action = agent.act(state_vector)

                    # Adjust reward based on action-outcome alignment
                    if action == 2:  # Buy
                        reward = price_return
                    elif action == 0:  # Sell
                        reward = -price_return
                    else:  # Hold
                        reward = abs(price_return) * 0.3  # Small positive reward for being right about direction

                    # Add small penalty for extreme actions to encourage some holding
                    if action != 1:  # Not hold
                        reward -= 0.001  # Small transaction cost/risk penalty

                    # Ensure reward is within reasonable bounds
                    reward = np.clip(reward, -0.1, 0.1)

                    if np.isnan(reward):
                        reward = 0.0
                except Exception as e:
                    print(f"[WARNING] Error calculating reward: {e}")
                    reward = 0.0

                # Record experience
                agent.remember(state_vector, action, reward, next_state_vector, False)
                experiences_collected += 1

            except Exception as e:
                print(f"[WARNING] Error in DQN experience collection sample {i}: {e}")
                continue

        print(f"[INFO] Collected {experiences_collected} experiences")

        # Train the agent if we have experiences
        if len(agent.memory) > 0:
            print(f"[INFO] Training DQN agent with {len(agent.memory)} experiences...")
            batch_size = min(128, len(agent.memory))

            # Number of training iterations based on memory size
            iterations = min(20, len(agent.memory) // 32)

            training_start = time.time()
            for _ in range(iterations):
                if time.time() - function_start_time > max_function_time * 0.75:
                    print("[WARNING] DQN training timeout reached")
                    break
                agent.replay(batch_size)

            print(f"[INFO] DQN training completed in {time.time() - training_start:.1f}s")

        # Get final recommendation using the new get_recommendation method
        recommendation = agent.get_recommendation(current_state)

        return recommendation

    except Exception as e:
        print(f"[ERROR] Error in DQN recommendation: {e}")
        traceback.print_exc()
        return 0.5  # Neutral score on failure


# Create Ensemble Prediction with log return components
def create_ensemble_prediction(momentum_score, reversion_score, lstm_prediction, dqn_recommendation,
                               volatility_data, market_regime, hurst_info, mean_reversion_info=None):
    """Create dynamically weighted ensemble with improved log return metrics"""

    # Base weights
    weights = {
        "momentum": 0.4,
        "reversion": 0.4,
        "lstm": 0.1,
        "dqn": 0.1
    }

    # Adjust weights based on volatility regime
    vol_regime = volatility_data.get("vol_regime", "Stable")
    if vol_regime == "Rising":
        # In rising volatility, favor mean reversion
        weights["momentum"] -= 0.1
        weights["reversion"] += 0.1
    elif vol_regime == "Falling":
        # In falling volatility, favor momentum
        weights["momentum"] += 0.1
        weights["reversion"] -= 0.1

    # Adjust weights based on market regime
    current_regime = market_regime.get("current_regime", "Unknown")
    if "High" in current_regime:
        # In high volatility regimes, increase ML model weights
        weights["lstm"] += 0.05
        weights["dqn"] += 0.05
        weights["momentum"] -= 0.05
        weights["reversion"] -= 0.05

    # Adjust based on Hurst exponent if available
    hurst_regime = hurst_info.get("regime", "Unknown")
    hurst_value = hurst_info.get("hurst", 0.5)

    # More precise adjustment based on hurst value
    if hurst_value < 0.3:  # Extremely strong mean reversion
        weights["reversion"] += 0.15
        weights["momentum"] -= 0.15
    elif hurst_value < 0.4:  # Strong mean reversion
        weights["reversion"] += 0.1
        weights["momentum"] -= 0.1
    elif hurst_value < 0.45:  # Moderate mean reversion
        weights["reversion"] += 0.05
        weights["momentum"] -= 0.05
    elif hurst_value > 0.7:  # Extremely strong trending
        weights["momentum"] += 0.15
        weights["reversion"] -= 0.15
    elif hurst_value > 0.6:  # Strong trending
        weights["momentum"] += 0.1
        weights["reversion"] -= 0.1
    elif hurst_value > 0.55:  # Moderate trending
        weights["momentum"] += 0.05
        weights["reversion"] -= 0.05

    # Adjust based on mean reversion half-life and beta if available
    if mean_reversion_info:
        half_life = mean_reversion_info.get("half_life", 0)
        beta = mean_reversion_info.get("beta", 0)

        # If strong mean reversion signal (negative beta, short half-life)
        if -1 < beta < -0.2 and 0 < half_life < 20:
            weights["reversion"] += 0.05
            weights["momentum"] -= 0.05
        # If no mean reversion (positive beta)
        elif beta > 0.1:
            weights["momentum"] += 0.05
            weights["reversion"] -= 0.05

    # Adjust based on volatility persistence if available
    vol_persistence = volatility_data.get("vol_persistence", 0.8)
    if vol_persistence > 0.9:  # High volatility persistence
        weights["reversion"] += 0.05
        weights["momentum"] -= 0.05
    elif vol_persistence < 0.6:  # Low volatility persistence
        weights["momentum"] += 0.03
        weights["reversion"] -= 0.03

    # Normalize weights to sum to 1
    total = sum(weights.values())
    for k in weights:
        weights[k] /= total

    # Calculate ensemble score
    ensemble_score = (
            weights["momentum"] * momentum_score +
            weights["reversion"] * (1 - reversion_score) +  # Invert reversion score (higher = more bearish)
            weights["lstm"] * lstm_prediction +
            weights["dqn"] * dqn_recommendation
    )

    return {
        "ensemble_score": ensemble_score,
        "weights": weights
    }


# Calculate sigma for a stock
# Enhanced sigma recommendation function with rich context
# Update the get_sigma_recommendation function to include new drawdown metrics
def get_sigma_recommendation(sigma, analysis_details):
    """Enhanced recommendation function with additional drawdown context"""
    # Get additional context for our recommendation
    momentum_score = analysis_details.get("momentum_score", 0.5)
    reversion_score = analysis_details.get("reversion_score", 0.5)
    recent_monthly_return = analysis_details.get("recent_monthly_return", 0)
    balance_factor = analysis_details.get("balance_factor", 0.5)
    hurst_regime = analysis_details.get("hurst_regime", "Unknown")
    mean_reversion_speed = analysis_details.get("mean_reversion_speed", "Unknown")
    mean_reversion_beta = analysis_details.get("mean_reversion_beta", 0)
    volatility_regime = analysis_details.get("volatility_regime", "Unknown")
    vol_persistence = analysis_details.get("vol_persistence", 0.8)
    market_regime = analysis_details.get("market_regime", "Unknown")
    max_drawdown = analysis_details.get("max_drawdown", 0)
    kelly = analysis_details.get("kelly", 0)
    sharpe = analysis_details.get("sharpe", 0)
    
    # Get enhanced drawdown metrics if available
    drawdown_analysis = analysis_details.get("drawdown_analysis", {})
    drawdown_count = drawdown_analysis.get("drawdown_count", 0)
    drawdown_probability = drawdown_analysis.get("drawdown_probability", 0)
    recovery_resilience = drawdown_analysis.get("recovery_resilience", 0.5)
    avg_drawdown_depth = drawdown_analysis.get("avg_drawdown_depth", 0)
    avg_recovery_time = drawdown_analysis.get("avg_recovery_time", 0)

    # Base recommendation on sigma
    if sigma > 0.8:
        base_rec = "STRONG BUY"
    elif sigma > 0.6:
        base_rec = "BUY"
    elif sigma > 0.4:
        base_rec = "HOLD"
    elif sigma > 0.2:
        base_rec = "SELL"
    else:
        base_rec = "STRONG SELL"

    # Add nuanced context based on recent performance, drawdowns, and advanced metrics
    if recent_monthly_return > 0.25 and sigma > 0.6:
        if "Mean Reversion" in hurst_regime and mean_reversion_speed in ["Fast", "Very Fast"]:
            context = f"Strong momentum with +{recent_monthly_return:.1%} monthly gain, but high mean reversion risk (Hurst={analysis_details.get('hurst_exponent', 0):.2f}, Beta={mean_reversion_beta:.2f})"
        else:
            context = f"Strong momentum with +{recent_monthly_return:.1%} monthly gain, elevated reversion risk but strong trend continues"
    elif recent_monthly_return > 0.15 and sigma > 0.6:
        if "Rising" in volatility_regime:
            context = f"Good momentum with +{recent_monthly_return:.1%} monthly gain but increasing volatility (persistence: {vol_persistence:.2f}), monitor closely"
        else:
            context = f"Good momentum with +{recent_monthly_return:.1%} monthly gain in stable volatility environment"
    elif recent_monthly_return > 0.10 and sigma > 0.6:
        if "Trending" in hurst_regime:
            context = f"Sustainable momentum with +{recent_monthly_return:.1%} monthly gain and strong trend characteristics (Hurst={analysis_details.get('hurst_exponent', 0):.2f})"
        else:
            context = f"Moderate momentum with +{recent_monthly_return:.1%} monthly gain showing balanced metrics"
    elif recent_monthly_return < -0.20 and sigma > 0.6:
        if "Mean Reversion" in hurst_regime:
            context = f"Strong reversal potential after {recent_monthly_return:.1%} monthly decline, log return metrics show bottoming pattern (Beta={mean_reversion_beta:.2f})"
        else:
            context = f"Potential trend change after {recent_monthly_return:.1%} decline but caution warranted"
    elif recent_monthly_return < -0.15 and sigma < 0.4:
        if "High" in market_regime:
            context = f"Continued weakness with {recent_monthly_return:.1%} monthly loss in high volatility regime"
        else:
            context = f"Negative trend with {recent_monthly_return:.1%} monthly loss and limited reversal signals"
    elif recent_monthly_return < -0.10 and sigma > 0.5:
        if mean_reversion_speed in ["Fast", "Very Fast"]:
            context = f"Potential rapid recovery after {recent_monthly_return:.1%} monthly decline (log reversion half-life: {analysis_details.get('mean_reversion_half_life', 0):.1f} days, Beta={mean_reversion_beta:.2f})"
        else:
            context = f"Potential stabilization after {recent_monthly_return:.1%} monthly decline, monitor for trend change"
    else:
        # Default context with advanced metrics, including drawdown analysis
        if momentum_score > 0.7 and "Trending" in hurst_regime:
            context = f"Strong trend characteristics (Hurst={analysis_details.get('hurst_exponent', 0):.2f}) with minimal reversal signals"
        elif momentum_score > 0.7 and reversion_score > 0.5:
            context = f"Strong but potentially overextended momentum in {volatility_regime} volatility regime (persistence: {vol_persistence:.2f})"
        elif momentum_score < 0.3 and "Mean Reversion" in hurst_regime:
            context = f"Strong mean-reverting characteristics (Hurst={analysis_details.get('hurst_exponent', 0):.2f}, Beta={mean_reversion_beta:.2f}) with weak momentum"
        elif momentum_score < 0.3 and reversion_score < 0.3:
            context = f"Weak directional signals in {market_regime} market regime"
        elif "High" in market_regime and "Rising" in volatility_regime:
            context = f"Mixed signals in high volatility environment - position sizing caution advised"
        elif abs(momentum_score - (1 - reversion_score)) < 0.1:
            context = f"Balanced indicators with no clear edge in {volatility_regime} volatility"
        else:
            context = f"Mixed signals requiring monitoring with log-based half-life of {analysis_details.get('mean_reversion_half_life', 0):.1f} days"
    
    # Add new enhanced drawdown analysis context
    if drawdown_count > 0:
        # Add drawdown specific context based on historical patterns
        if drawdown_count >= 3 and avg_drawdown_depth < -0.15:
            context += f" | History of frequent significant drawdowns ({drawdown_count} events, avg depth: {avg_drawdown_depth:.1%})"
        
        if drawdown_probability > 0.6:
            context += f" | High probability of future drawdown ({drawdown_probability:.2f}) based on historical patterns"
        elif drawdown_probability > 0.3:
            context += f" | Moderate drawdown risk ({drawdown_probability:.2f})"
            
        if recovery_resilience > 0.7:
            context += f" | Strong recovery resilience ({recovery_resilience:.2f})"
        elif recovery_resilience < 0.3:
            context += f" | Poor historical recovery pattern ({recovery_resilience:.2f})"
            
        if avg_recovery_time and avg_recovery_time > 60:
            context += f" | Typically long recovery periods ({avg_recovery_time:.0f} days avg)"
    
    # Add risk metrics
    if max_drawdown < -0.4:
        context += f" | High historical drawdown risk ({max_drawdown:.1%})"
    
    if kelly < -0.2:
        context += f" | Negative expectancy (Kelly={kelly:.2f})"
    elif kelly > 0.3:
        context += f" | Strong positive expectancy (Kelly={kelly:.2f})"
    
    # Add Sharpe ratio if available
    if sharpe > 1.5:
        context += f" | Excellent risk-adjusted returns (Sharpe={sharpe:.2f})"
    elif sharpe < 0:
        context += f" | Poor risk-adjusted returns (Sharpe={sharpe:.2f})"
    
    # Combine base recommendation with context
    recommendation = f"{base_rec} - {context}"
    
    return recommendation

# Generate price predictions based on analysis results
def generate_optimized_price_predictions(data, analysis_details, forecast_days=60, num_paths=2000):
    """
    Generate optimized price predictions with narrower, more realistic bands while
    preserving the integrity of the sigma value and analysis structure.
    
    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing historical price data
    analysis_details: dict
        Dictionary with analysis details
    forecast_days: int
        Number of days to forecast
    num_paths: int
        Number of Monte Carlo simulation paths

    Returns:
    --------
    dict
        Dictionary with prediction results
    """
    try:
        # Get current price
        price_col = 'close' if 'close' in data.columns else '4. close'
        current_price = data[price_col].iloc[-1]

        # Extract key factors from analysis
        analysis_factors = {
            'sigma': analysis_details.get('sigma', 0.5),
            'raw_sigma': analysis_details.get('raw_sigma', 0.5),
            'momentum_score': analysis_details.get("momentum_score", 0.5),
            'reversion_score': analysis_details.get("reversion_score", 0.5),
            'hurst_exponent': analysis_details.get("hurst_exponent", 0.5),
            'market_regime': analysis_details.get("market_regime", "Unknown"),
            'volatility_regime': analysis_details.get("volatility_regime", "Stable"),
            'max_drawdown': analysis_details.get("max_drawdown", 0),
            'mean_reversion_half_life': analysis_details.get("mean_reversion_half_life", 0),
            'mean_reversion_beta': analysis_details.get("mean_reversion_beta", 0),
            'sharpe': analysis_details.get("sharpe", 0),
            'kelly': analysis_details.get("kelly", 0)
        }

        # Use improved adaptive volatility function for more accurate volatility estimation
        hist_volatility = calculate_adaptive_volatility(data)
        print(f"[INFO] Using adaptive volatility for prediction bands: {hist_volatility:.4f}")

        # Apply volatility adjustment based on analysis
        vol_adjustment = 1.0
        
        # Adjust volatility based on Hurst exponent and mean reversion characteristics
        if analysis_factors['hurst_exponent'] < 0.4:  # Strong mean reversion
            vol_adjustment *= 0.9  # Reduce volatility for strong mean-reverting assets
            print(f"[INFO] Reducing volatility by 10% due to strong mean reversion (Hurst: {analysis_factors['hurst_exponent']:.2f})")
        elif analysis_factors['hurst_exponent'] > 0.6:  # Strong trending
            # For trending assets, maintain higher volatility but improve predictability
            print(f"[INFO] Maintaining volatility for trending asset (Hurst: {analysis_factors['hurst_exponent']:.2f})")
            
        # Adjust based on mean reversion speed
        if 0 < analysis_factors['mean_reversion_half_life'] < 20:  # Fast mean reversion
            vol_adjustment *= 0.9
            print(f"[INFO] Reducing volatility by 10% due to fast mean reversion (Half-life: {analysis_factors['mean_reversion_half_life']:.1f} days)")
            
        # Apply regime-based adjustment
        if analysis_factors['volatility_regime'] == "Rising":
            vol_adjustment *= 1.1
            print(f"[INFO] Increasing volatility by 10% due to rising volatility regime")
        elif analysis_factors['volatility_regime'] == "Falling":
            vol_adjustment *= 0.9
            print(f"[INFO] Reducing volatility by 10% due to falling volatility regime")
            
        # Apply drawdown-based adjustment if available
        if 'drawdown_analysis' in analysis_details:
            drawdown_analysis = analysis_details['drawdown_analysis']
            risk_adjustment = calculate_adaptive_risk_adjustment(
                drawdown_analysis.get('max_drawdown', 0),
                drawdown_analysis.get('drawdown_frequency', 0),
                drawdown_analysis.get('drawdown_severity', 0),
                drawdown_analysis.get('recovery_resilience', 0.5)
            )
            vol_adjustment *= risk_adjustment
            print(f"[INFO] Applied drawdown-based risk adjustment: {risk_adjustment:.2f}")
            
        # Final adjusted volatility
        adjusted_volatility = hist_volatility * vol_adjustment
        print(f"[INFO] Final adjusted volatility: {adjusted_volatility:.4f} (original: {hist_volatility:.4f})")

        # Calculate expected return based on sigma with more realistic boundaries
        # Use a more conservative range: -20% to +20% annual return instead of -25% to 25%
        expected_annual_return = (analysis_factors['sigma'] - 0.5) * 0.4
        
        # Adjust expected return based on Sharpe and Kelly
        if analysis_factors['sharpe'] < 0:
            expected_annual_return *= max(0.5, 1 + analysis_factors['sharpe'])  # Reduce expectations for negative Sharpe
            print(f"[INFO] Tempering return expectations due to negative Sharpe: {analysis_factors['sharpe']:.2f}")
            
        if analysis_factors['kelly'] < -0.1:
            expected_annual_return *= max(0.7, 1 + analysis_factors['kelly'])  # Reduce expectations for negative Kelly
            print(f"[INFO] Tempering return expectations due to negative Kelly: {analysis_factors['kelly']:.2f}")
            
        # Ensure return expectation is reasonable based on sigma
        if analysis_factors['sigma'] > 0.8 and expected_annual_return < 0.1:
            expected_annual_return = 0.1  # Minimum 10% expected return for very high sigma
        elif analysis_factors['sigma'] < 0.2 and expected_annual_return > -0.1:
            expected_annual_return = -0.1  # Maximum -10% expected return for very low sigma
            
        print(f"[INFO] Expected annual return after adjustments: {expected_annual_return:.2%}")

        # Calculate daily expected return
        daily_return = (1 + expected_annual_return) ** (1 / 252) - 1

        # Create date range for forecast
        last_date = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.today()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

        # Initialize arrays for price paths
        daily_vol = adjusted_volatility / np.sqrt(252)
        price_paths = np.zeros((num_paths, forecast_days))
        
        # Generate all random components at once for efficiency
        # Use truncated normal distribution to avoid extreme tail events
        random_components = np.random.normal(0, daily_vol, (num_paths, forecast_days))
        
        # Clip extreme values to reduce fat tails (limit to 2.5 standard deviations)
        # This is key to reducing unrealistic prediction spreads
        random_components = np.clip(random_components, -2.5 * daily_vol, 2.5 * daily_vol)

        # Vectorized first day calculation
        price_paths[:, 0] = current_price * (1 + daily_return + random_components[:, 0])

        # Define trend line - use expected growth path as the center line
        trend_line = np.zeros(forecast_days)
        trend_line[0] = current_price
        for i in range(1, forecast_days):
            trend_line[i] = trend_line[i-1] * (1 + daily_return)

        # Calculate mean reversion strength based on Hurst exponent and beta
        mean_reversion_strength = max(0, 1.0 - analysis_factors['hurst_exponent'])
        
        # Boost mean reversion if beta is significantly negative
        if analysis_factors['mean_reversion_beta'] < -0.3:
            mean_reversion_strength += abs(analysis_factors['mean_reversion_beta']) * 0.2
            
        # Cap mean reversion strength at a reasonable level
        mean_reversion_strength = min(0.8, mean_reversion_strength)
        print(f"[INFO] Mean reversion strength: {mean_reversion_strength:.2f}")

        # Vectorized simulation for remaining days
        for j in range(1, forecast_days):
            # Apply drift to all paths
            drift = daily_return
            
            # Apply random component to all paths
            random_component = random_components[:, j]
            
            # Calculate mean reversion forces
            if j > 2:  # Start mean reversion early
                # Calculate distance from trend for all paths
                distance = price_paths[:, j - 1] / trend_line[j - 1] - 1
                
                # Apply adaptive mean reversion that strengthens as paths diverge
                # This is critical for reducing unrealistic spreads
                adaptive_reversion = np.where(
                    abs(distance) > 0.05,  # For paths that have diverged more than 5%
                    mean_reversion_strength * (1 + np.minimum(abs(distance) * 5, 1.5)),  # Increase reversion force
                    mean_reversion_strength
                )
                
                # Scale mean reversion effect with time
                time_factor = 0.05 + (j / forecast_days) * 0.1
                mean_reversion = -distance * adaptive_reversion * time_factor
                
                # Limit extreme mean reversion for stability
                mean_reversion = np.clip(mean_reversion, -0.03, 0.03)
            else:
                mean_reversion = np.zeros(num_paths)
            
            # Generate next price for all paths with these dynamics
            price_paths[:, j] = price_paths[:, j - 1] * (1 + drift + random_component + mean_reversion)
        
        # Filter unrealistic paths to improve prediction quality
        filtered_paths = filter_unrealistic_paths(
            price_paths, 
            trend_line,
            max_daily_change=0.08,  # Maximum 8% daily change
            max_trend_deviation=0.4  # Maximum 40% deviation from trend
        )
        
        # Ensure we have enough paths after filtering
        if len(filtered_paths) < num_paths * 0.7:
            print(f"[INFO] Many paths filtered out ({len(filtered_paths)} remaining). Resampling to {num_paths} paths.")
            filtered_paths = resample_to_original_size(filtered_paths, num_paths)
        
        # Use filtered paths for statistics
        price_paths_final = filtered_paths
        
        # Calculate statistics on paths
        mean_path = np.mean(price_paths_final, axis=0)
        
        # Use percentiles for confidence bands
        # Using 5-95 instead of 2.5-97.5 for tighter 90% confidence
        lower_bound_95 = np.percentile(price_paths_final, 5, axis=0)
        upper_bound_95 = np.percentile(price_paths_final, 95, axis=0)
        
        # 68% confidence bands (1 standard deviation)
        lower_bound_68 = np.percentile(price_paths_final, 16, axis=0)
        upper_bound_68 = np.percentile(price_paths_final, 84, axis=0)

        # Calculate price targets and returns
        price_target_30d = mean_path[min(29, forecast_days - 1)]
        price_target_60d = mean_path[min(59, forecast_days - 1)]

        # Calculate returns
        expected_return_30d = (price_target_30d / current_price - 1) * 100
        expected_return_60d = (price_target_60d / current_price - 1) * 100

        # Return prediction results
        return {
            "current_price": current_price,
            "forecast_dates": forecast_dates,
            "mean_path": mean_path,
            "lower_bound_95": lower_bound_95,
            "upper_bound_95": upper_bound_95,
            "lower_bound_68": lower_bound_68,
            "upper_bound_68": upper_bound_68,
            "price_target_30d": price_target_30d,
            "price_target_60d": price_target_60d,
            "expected_return_30d": expected_return_30d,
            "expected_return_60d": expected_return_60d,
            "hist_volatility": hist_volatility,
            "adjusted_volatility": adjusted_volatility,
            "expected_annual_return": expected_annual_return,
            "trend_line": trend_line,
            "sigma": analysis_factors['sigma'],
            "num_paths": len(price_paths_final)
        }
    except Exception as e:
        print(f"[ERROR] Error generating optimized price predictions: {e}")
        traceback.print_exc()
        return None

def calculate_adaptive_volatility(data, lookback_short=10, lookback_med=30, lookback_long=180):
    """
    Calculate adaptive volatility that weights recent periods more heavily and 
    adjusts based on volatility regime changes.
    
    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data and volatility metrics
    lookback_short: int
        Short lookback period (days)
    lookback_med: int
        Medium lookback period (days)
    lookback_long: int
        Long lookback period (days)
        
    Returns:
    --------
    float
        Adaptive volatility estimate
    """
    # Determine which volatility measure to use (prefer log_volatility)
    if 'log_volatility' in data.columns:
        vol_short = data['log_volatility'].iloc[-lookback_short:].mean() * np.sqrt(252)
        vol_med = data['log_volatility'].iloc[-lookback_med:].mean() * np.sqrt(252)
        vol_long = data['log_volatility'].iloc[-lookback_long:].mean() * np.sqrt(252)
        print(f"[INFO] Using log volatility for adaptive estimation: short={vol_short:.4f}, med={vol_med:.4f}, long={vol_long:.4f}")
    elif 'volatility' in data.columns:
        vol_short = data['volatility'].iloc[-lookback_short:].mean() * np.sqrt(252)
        vol_med = data['volatility'].iloc[-lookback_med:].mean() * np.sqrt(252)
        vol_long = data['volatility'].iloc[-lookback_long:].mean() * np.sqrt(252)
        print(f"[INFO] Using standard volatility for adaptive estimation")
    else:
        # Calculate from price data
        price_col = 'close' if 'close' in data.columns else '4. close'
        prices = data[price_col].values
        log_returns = np.diff(np.log(prices))
        
        # Calculate for different lookback periods
        vol_short = np.std(log_returns[-min(lookback_short, len(log_returns)):]) * np.sqrt(252)
        vol_med = np.std(log_returns[-min(lookback_med, len(log_returns)):]) * np.sqrt(252)
        vol_long = np.std(log_returns[-min(lookback_long, len(log_returns)):]) * np.sqrt(252)
    
    # Detect volatility regime
    vol_trend = "Stable"
    if vol_short > vol_med * 1.2:
        vol_trend = "Rising"
    elif vol_short < vol_med * 0.8:
        vol_trend = "Falling"
    
    # Combine volatilities with exponential weighting (more emphasis on recent data)
    # but with safeguards against extreme short-term volatility
    if vol_trend == "Rising":
        # When volatility is rising, increase weight of medium-term to avoid overreaction
        # but still give more weight to recent data
        weights = [0.4, 0.4, 0.2]  # [short, med, long]
    elif vol_trend == "Falling":
        # When volatility is falling, give more weight to short-term
        weights = [0.5, 0.3, 0.2]
    else:
        # In stable regime, balance between recent and historical
        weights = [0.4, 0.35, 0.25]
    
    adaptive_vol = weights[0] * vol_short + weights[1] * vol_med + weights[2] * vol_long
    
    # Apply dampening factor to extreme values
    if adaptive_vol > vol_long * 1.5:
        # Dampen extremely high volatility estimates
        dampened_vol = vol_long * 1.5 * 0.7 + adaptive_vol * 0.3
        print(f"[INFO] Dampening high volatility: {adaptive_vol:.4f} → {dampened_vol:.4f}")
        adaptive_vol = dampened_vol
    
    # Evaluate market conditions for further adjustments
    if vol_trend == "Rising" and vol_short > vol_long * 1.3:
        # In rapidly rising volatility environments, slightly reduce prediction bands
        adjustment_factor = 0.9
        print(f"[INFO] Applying dampening factor of {adjustment_factor} due to rapidly rising volatility")
        adaptive_vol *= adjustment_factor
    
    return adaptive_vol

def filter_unrealistic_paths(price_paths, trend_line, max_daily_change=0.1, max_trend_deviation=0.5):
    """
    Filter out unrealistic paths from Monte Carlo simulation results.
    
    Parameters:
    -----------
    price_paths: numpy.ndarray
        Array of price paths from Monte Carlo simulation
    trend_line: numpy.ndarray
        Expected trend line (central tendency path)
    max_daily_change: float
        Maximum allowed daily percentage change
    max_trend_deviation: float
        Maximum allowed deviation from trend line
        
    Returns:
    --------
    numpy.ndarray
        Filtered price paths
    """
    num_paths, forecast_days = price_paths.shape
    valid_paths = np.ones(num_paths, dtype=bool)
    
    # Check for unrealistic daily changes
    for i in range(num_paths):
        for j in range(1, forecast_days):
            daily_change = abs(price_paths[i, j] / price_paths[i, j-1] - 1)
            if daily_change > max_daily_change:
                valid_paths[i] = False
                break
    
    # Check for excessive deviation from trend
    for i in range(num_paths):
        if not valid_paths[i]:
            continue
            
        max_deviation = np.max(np.abs(price_paths[i, :] / trend_line - 1))
        if max_deviation > max_trend_deviation:
            valid_paths[i] = False
    
    # Return filtered paths
    filtered_paths = price_paths[valid_paths, :]
    
    # If too many paths were filtered, replace with resampled paths
    if len(filtered_paths) < num_paths * 0.7:
        print(f"[INFO] Resampling {num_paths - len(filtered_paths)} paths that were filtered out")
        # Resample from valid paths
        resampled_paths = np.zeros_like(price_paths)
        resampled_paths[:len(filtered_paths)] = filtered_paths
        
        # For remaining paths, sample from valid paths with small random variations
        valid_indices = np.where(valid_paths)[0]
        for i in range(len(filtered_paths), num_paths):
            # Select a random valid path
            idx = np.random.choice(valid_indices)
            # Apply small random variations
            variation = np.random.normal(0, 0.01, forecast_days)
            resampled_paths[i] = price_paths[idx] * (1 + variation)
        
        return resampled_paths
    else:
        return filtered_paths

def resample_to_original_size(filtered_paths, original_size):
    """
    Resample filtered paths to match the original size.
    
    Parameters:
    -----------
    filtered_paths: numpy.ndarray
        Array of filtered price paths
    original_size: int
        Original number of paths
        
    Returns:
    --------a
    numpy.ndarray
        Resampled price paths with original size
    """
    if len(filtered_paths) == original_size:
        return filtered_paths
        
    # If we have too few paths after filtering
    if len(filtered_paths) < original_size:
        # Resample with replacement to get back to original size
        indices = np.random.choice(len(filtered_paths), original_size - len(filtered_paths), replace=True)
        additional_paths = filtered_paths[indices]
        
        # Add small random variations to avoid exact duplicates
        variation = np.random.normal(0, 0.005, additional_paths.shape)
        additional_paths = additional_paths * (1 + variation)
        
        # Combine original filtered paths with additional ones
        resampled_paths = np.vstack([filtered_paths, additional_paths])
        return resampled_paths
    else:
        # If we have too many paths, sample down to original size
        indices = np.random.choice(len(filtered_paths), original_size, replace=False)
        return filtered_paths[indices]

# Function to calculate risk adjustment factor based on drawdown characteristics
def calculate_adaptive_risk_adjustment(max_drawdown, drawdown_frequency, drawdown_severity, recovery_resilience):
    """
    Calculate an adaptive risk adjustment factor based on drawdown characteristics
    
    Parameters:
    -----------
    max_drawdown: float 
        Maximum drawdown (negative number)
    drawdown_frequency: float
        How often drawdowns occur (higher = more frequent)
    drawdown_severity: float
        Combined measure of depth and duration
    recovery_resilience: float
        Measure of recovery speed (0-1, higher = faster recovery)
        
    Returns:
    --------
    float
        Risk adjustment factor (0-1)
    """
    # Base adjustment - lower for stocks with severe drawdowns
    base_adjustment = 1.0
    
    # Adjust for max drawdown depth
    if max_drawdown < -0.5:  # More than 50% drawdown
        base_adjustment *= 0.7
    elif max_drawdown < -0.3:  # 30-50% drawdown
        base_adjustment *= 0.8
    elif max_drawdown < -0.2:  # 20-30% drawdown
        base_adjustment *= 0.9
    
    # Adjust for frequency (more frequent = lower factor)
    freq_adjustment = max(0.7, 1.0 - (drawdown_frequency * 0.05))
    
    # Adjust for severity
    severity_adjustment = max(0.7, 1.0 - (drawdown_severity * 0.1))
    
    # Adjust for recovery resilience (higher = better)
    resilience_adjustment = 0.7 + (recovery_resilience * 0.3)
    
    # Combine all factors
    risk_adjustment = base_adjustment * freq_adjustment * severity_adjustment * resilience_adjustment
    
    # Ensure result is reasonable (0.5-1.0 range)
    return max(0.5, min(1.0, risk_adjustment))

# Create prediction plot
def create_improved_prediction_plot(stock_data, prediction_data, symbol, plot_dir="prediction_plots"):
    """
    Create improved stock price prediction plot with tighter bands and more information.
    
    Parameters:    
    -----------
    stock_data: pandas DataFrame
        DataFrame containing historical price data
    prediction_data: dict
        Dictionary with prediction results
    symbol: str
        Stock symbol for the plot title
    plot_dir: str
        Directory to save the plot
        
    Returns:
    --------
    str
        Path to the saved plot
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(plot_dir, exist_ok=True)
        print(f"[INFO] Ensuring directory exists for prediction plots: {plot_dir}")

        # Get the current timestamp for the filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_filename = f"{symbol}_prediction_{timestamp}.png"
        plot_path = os.path.join(plot_dir, plot_filename)

        # Create plot with historical and predicted data
        plt.figure(figsize=(12, 8))

        # Get price column
        price_col = 'close' if 'close' in stock_data.columns else '4. close'

        # Get historical dates and prices for plotting
        if isinstance(stock_data.index, pd.DatetimeIndex):
            hist_dates = stock_data.index[-120:]  # Last 120 days of history
            hist_prices = stock_data[price_col][-120:]
        else:
            # If index isn't datetime, create placeholder dates
            hist_dates = pd.date_range(end=pd.Timestamp.today(), periods=min(120, len(stock_data)))
            hist_prices = stock_data[price_col][-min(120, len(stock_data)):]

        # Plot historical data
        plt.plot(hist_dates, hist_prices, label="Historical Price", color='blue', linewidth=2)

        # Extract prediction data
        forecast_dates = prediction_data['forecast_dates']
        mean_path = prediction_data['mean_path']
        lower_bound_95 = prediction_data['lower_bound_95']
        upper_bound_95 = prediction_data['upper_bound_95']
        lower_bound_68 = prediction_data['lower_bound_68']
        upper_bound_68 = prediction_data['upper_bound_68']
        
        num_paths = prediction_data.get('num_paths', 180)

        # Add trend line if available
        if 'trend_line' in prediction_data:
            trend_line = prediction_data['trend_line']
            plt.plot(forecast_dates, trend_line, '--', label="Expected Trend", color='gray', linewidth=1, alpha=0.7)

        # Plot mean forecast with improved styling
        plt.plot(forecast_dates, mean_path, label="Mean Forecast", color='green', linewidth=2.5)

        # Plot confidence intervals with improved clarity
        plt.fill_between(forecast_dates, lower_bound_95, upper_bound_95, color='green', alpha=0.1,
                         label="90% Confidence")  # Using 5-95% instead of 2.5-97.5%
        plt.fill_between(forecast_dates, lower_bound_68, upper_bound_68, color='green', alpha=0.25,
                         label="68% Confidence")

        # Add price targets
        price_target_30d = prediction_data['price_target_30d']
        price_target_60d = prediction_data['price_target_60d']
        expected_return_30d = prediction_data['expected_return_30d']
        expected_return_60d = prediction_data['expected_return_60d']

        # Mark 30-day and 60-day targets more clearly
        if len(forecast_dates) >= 30:
            plt.plot(forecast_dates[29], price_target_30d, 'o', color='purple', markersize=10, zorder=5)
            plt.annotate(f"30d: ${price_target_30d:.2f} ({expected_return_30d:.1f}%)",
                         (forecast_dates[29], price_target_30d),
                         xytext=(10, 15), textcoords='offset points',
                         fontsize=12, fontweight='bold', color='purple',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="purple", alpha=0.8))

        if len(forecast_dates) >= 60:
            plt.plot(forecast_dates[59], price_target_60d, 'o', color='red', markersize=10, zorder=5)
            plt.annotate(f"60d: ${price_target_60d:.2f} ({expected_return_60d:.1f}%)",
                         (forecast_dates[59], price_target_60d),
                         xytext=(10, -25), textcoords='offset points',
                         fontsize=12, fontweight='bold', color='red',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

        # Add key support/resistance levels based on the 68% confidence bands
        support_30d = lower_bound_68[29] if len(forecast_dates) >= 30 else None
        resistance_30d = upper_bound_68[29] if len(forecast_dates) >= 30 else None
        
        if support_30d and resistance_30d:
            plt.axhline(y=support_30d, color='blue', linestyle='--', alpha=0.4)
            plt.axhline(y=resistance_30d, color='red', linestyle='--', alpha=0.4)
            
            plt.annotate(f"Support: ${support_30d:.2f}",
                         (forecast_dates[15], support_30d),
                         xytext=(5, -20), textcoords='offset points',
                         fontsize=10, color='blue')
            
            plt.annotate(f"Resistance: ${resistance_30d:.2f}",
                         (forecast_dates[15], resistance_30d),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=10, color='red')

        # Add title and labels with improved information
        current_price = prediction_data['current_price']
        volatility = prediction_data['adjusted_volatility']
        annual_return = prediction_data['expected_annual_return']
        
        # More informative title with risk assessment
        volatility_level = "High" if volatility > 0.35 else "Moderate" if volatility > 0.2 else "Low"
        
        plt.title(
            f"{symbol} Price Forecast\n"
            f"Current: ${current_price:.2f} | Expected Annual Return: {annual_return * 100:.1f}% | {volatility_level} Volatility: {volatility * 100:.1f}%",
            fontsize=14, fontweight='bold')
        
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price ($)", fontsize=12)

        # Format x-axis to show dates properly
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45)

        # Add grid and legend with better placement
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left", framealpha=0.9)
        
        # Add sigma score and confidence information
        sigma = prediction_data.get('sigma', 0.5)
        sigma_text = "Bullish" if sigma > 0.7 else "Bearish" if sigma < 0.3 else "Neutral"
        confidence_width = ((upper_bound_68[-1] - lower_bound_68[-1]) / mean_path[-1]) * 100
        
        info_text = (
            f"Sigma Score: {sigma:.2f} ({sigma_text})\n"
            f"Confidence Width: ±{confidence_width/2:.1f}%\n"
            f"Prediction Based on {num_paths} Simulations"
        )
        
        # Add text box with additional information
        plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        # Adjust layout and save with higher quality
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Improved price prediction plot saved to {plot_path}")
        return plot_path

    except Exception as e:
        print(f"[ERROR] Error creating improved prediction plot: {e}")
        traceback.print_exc()
        return None

def append_stock_result(result):
    """
    Append detailed stock analysis result to the output file
    
    Parameters:
    -----------
    result: dict
        Stock analysis result
    """
    try:
        with open(OUTPUT_FILE, "a") as file:
            # Basic information
            file.write(f"=== ANALYSIS FOR {result['symbol']} ===\n")
            
            # Add company info if available
            if result.get('company_info'):
                company = result['company_info']
                file.write(f"Company: {company.get('Name', 'N/A')}\n")
                file.write(f"Industry: {company.get('Industry', 'N/A')}\n")
                file.write(f"Sector: {company.get('Sector', 'N/A')}\n")
            
            # Price and changes
            if result.get('quote_data'):
                quote = result['quote_data']
                file.write(f"Current Price: ${quote.get('price', 0):.2f}\n")
                file.write(f"Change: {quote.get('change', 0):.2f} ({quote.get('change_percent', '0%')})\n")
            else:
                file.write(f"Current Price: ${result['price']:.2f}\n")
            
            file.write(f"Sigma Score: {result['sigma']:.5f}\n")
            file.write(f"Recommendation: {result['recommendation']}\n\n")
            
            # Detailed analysis
            analysis = result['analysis']
            
            file.write("--- COMPONENT SCORES ---\n")
            file.write(f"Momentum Score: {analysis.get('momentum_score', 0):.3f}\n")
            file.write(f"Reversion Score: {analysis.get('reversion_score', 0):.3f}\n")
            file.write(f"Balance Factor: {analysis.get('balance_factor', 0):.3f}\n")
            
            file.write("\n--- TECHNICAL INDICATORS ---\n")
            file.write(f"RSI: {analysis.get('rsi', 0):.2f}\n")
            file.write(f"MACD: {analysis.get('macd', 0):.5f}\n")
            file.write(f"SMA Trend: {analysis.get('sma_trend', 0):.5f}\n")
            file.write(f"Distance from SMA200: {analysis.get('dist_from_sma200', 0):.3f}\n")
            file.write(f"Volatility: {analysis.get('volatility', 0):.5f}\n")
            
            file.write("\n--- MARKET REGIME ---\n")
            file.write(f"Hurst Exponent: {analysis.get('hurst_exponent', 0):.3f} ({analysis.get('hurst_regime', 'Unknown')})\n")
            file.write(f"Mean Reversion Half-Life: {analysis.get('mean_reversion_half_life', 0):.1f} days ({analysis.get('mean_reversion_speed', 'Unknown')})\n")
            file.write(f"Mean Reversion Beta: {analysis.get('mean_reversion_beta', 0):.3f}\n")
            file.write(f"Volatility Regime: {analysis.get('volatility_regime', 'Unknown')}\n")
            file.write(f"Volatility Term Structure: {analysis.get('vol_term_structure', 0):.3f}\n")
            file.write(f"Volatility Persistence: {analysis.get('vol_persistence', 0):.3f}\n")
            file.write(f"Market Regime: {analysis.get('market_regime', 'Unknown')}\n")
            
            file.write("\n--- RISK METRICS ---\n")
            file.write(f"Maximum Drawdown: {analysis.get('max_drawdown', 0):.2%}\n")
            file.write(f"Kelly Criterion: {analysis.get('kelly', 0):.3f}\n")
            file.write(f"Sharpe Ratio: {analysis.get('sharpe', 0):.3f}\n")
            
            file.write("\n--- ADVANCED METRICS ---\n")
            if 'advanced_metrics' in analysis:
                advanced = analysis['advanced_metrics']
                for key, value in advanced.items():
                    if isinstance(value, dict):
                        file.write(f"{key}:\n")
                        for subkey, subvalue in value.items():
                            file.write(f"  {subkey}: {subvalue}\n")
                    else:
                        file.write(f"{key}: {value}\n")
            else:
                file.write("No advanced metrics available\n")
            
            file.write("\n--- MACHINE LEARNING ---\n")
            file.write(f"LSTM Prediction: {analysis.get('lstm_prediction', 0):.3f}\n")
            file.write(f"DQN Recommendation: {analysis.get('dqn_recommendation', 0):.3f}\n")
            
            # Add more detailed metrics if available
            if 'multifractal' in analysis:
                file.write("\n--- MULTIFRACTAL ANALYSIS ---\n")
                for key, value in analysis['multifractal'].items():
                    if isinstance(value, dict):
                        file.write(f"{key}:\n")
                        for subkey, subvalue in value.items():
                            file.write(f"  {subkey}: {subvalue}\n")
                    else:
                        file.write(f"{key}: {value}\n")
                        
            if 'tail_risk' in analysis:
                file.write("\n--- TAIL RISK ANALYSIS ---\n")
                tail_risk = analysis['tail_risk']
                if isinstance(tail_risk, dict):
                    # Extract key metrics
                    if 'tail_type' in tail_risk:
                        file.write(f"Tail Type: {tail_risk['tail_type']}\n")
                    if 'tail_description' in tail_risk:
                        file.write(f"Description: {tail_risk['tail_description']}\n")
                    if 'expected_shortfall' in tail_risk:
                        es = tail_risk['expected_shortfall']
                        for key, value in es.items():
                            file.write(f"{key}: {value:.2%}\n")
                    
            if 'wavelet' in analysis:
                file.write("\n--- WAVELET ANALYSIS ---\n")
                wavelet = analysis['wavelet']
                if isinstance(wavelet, dict) and 'wavelet_transform' in wavelet:
                    wt = wavelet['wavelet_transform']
                    if 'dominant_period' in wt:
                        file.write(f"Dominant Cycle: {wt['dominant_period']:.2f} days\n")
                    if 'dominant_frequency' in wt:
                        file.write(f"Dominant Frequency: {wt['dominant_frequency']:.6f}\n")
            
            # Add fundamental data if available
            if result.get('company_info'):
                file.write("\n--- FUNDAMENTAL DATA ---\n")
                fund_data = result['company_info']
                metrics = [
                    ('MarketCapitalization', 'Market Cap', ''),
                    ('PERatio', 'P/E Ratio', ''),
                    ('PEGRatio', 'PEG Ratio', ''),
                    ('PriceToBookRatio', 'P/B Ratio', ''),
                    ('EVToEBITDA', 'EV/EBITDA', ''),
                    ('ProfitMargin', 'Profit Margin', '%'),
                    ('OperatingMarginTTM', 'Operating Margin', '%'),
                    ('ReturnOnAssetsTTM', 'ROA', '%'),
                    ('ReturnOnEquityTTM', 'ROE', '%'),
                    ('RevenueTTM', 'Revenue TTM', ''),
                    ('GrossProfitTTM', 'Gross Profit TTM', ''),
                    ('DilutedEPSTTM', 'EPS TTM', ''),
                    ('QuarterlyEarningsGrowthYOY', 'Quarterly Earnings Growth', '%'),
                    ('QuarterlyRevenueGrowthYOY', 'Quarterly Revenue Growth', '%'),
                    ('AnalystTargetPrice', 'Analyst Target', '$'),
                    ('Beta', 'Beta', ''),
                    ('52WeekHigh', '52-Week High', '$'),
                    ('52WeekLow', '52-Week Low', '$'),
                    ('50DayMovingAverage', '50-Day MA', '$'),
                    ('200DayMovingAverage', '200-Day MA', '$'),
                    ('DividendYield', 'Dividend Yield', '%'),
                    ('DividendPerShare', 'Dividend Per Share', '$'),
                    ('PayoutRatio', 'Payout Ratio', '%'),
                ]
                
                for key, label, suffix in metrics:
                    if key in fund_data and fund_data[key]:
                        try:
                            # Format numbers properly
                            if key in ['MarketCapitalization', 'RevenueTTM', 'GrossProfitTTM']:
                                # Convert large numbers to billions/millions
                                value = float(fund_data[key])
                                if value >= 1e9:
                                    formatted = f"${value/1e9:.2f}B"
                                elif value >= 1e6:
                                    formatted = f"${value/1e6:.2f}M"
                                else:
                                    formatted = f"${value:.2f}"
                            elif suffix == '$':
                                formatted = f"${float(fund_data[key]):.2f}"
                            elif suffix == '%':
                                formatted = f"{float(fund_data[key]):.2f}%"
                            else:
                                formatted = f"{fund_data[key]}"
                                
                            file.write(f"{label}: {formatted}\n")
                        except:
                            file.write(f"{label}: {fund_data[key]}\n")
            
            file.write("\n" + "="*50 + "\n\n")
            
            return True
    except Exception as e:
        print(f"[ERROR] Failed to append result: {e}")
        traceback.print_exc()
        return False

def initialize_output_file():
    """Initialize the output file with a header"""
    try:
        # Create directory for the output file if needed
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Created directory for output file: {output_dir}")

        # Create or append to the output file
        mode = "w" # Always write, overwriting existing file
        with open(OUTPUT_FILE, mode) as file:
            # Header will always be written since mode is 'w'
            file.write("===== OPTIMIZED STOCK ANALYSIS RESULTS =====\n")
            file.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(
                f"GPU Acceleration: {'Enabled' if len(tensorflow.config.list_physical_devices('GPU')) > 0 else 'Disabled'}\n")
            file.write("=" * 50 + "\n\n")

        print(f"[INFO] Output file initialized: {OUTPUT_FILE}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize output file: {e}")
        traceback.print_exc()
        return False

def perform_analysis_for_server(symbol, user_id=None):
    """
    Performs stock analysis for a given symbol, optionally for a user.
    Initializes client, analyzes stock, initializes output file, appends result, and prints summary.
    """
    # Create prediction plots directory if it doesn't exist
    # This is a good place as analyze_stock (which might create a plot) is called soon after.
    try:
        os.makedirs("prediction_plots", exist_ok=True)
        # print("[INFO] Ensured directory exists for prediction plots: prediction_plots") # Optional: can be verbose
    except Exception as e:
        print(f"[WARNING] Failed to create prediction plots directory: {e}")
        traceback.print_exc()

    client = AlphaVantageClient(ALPHA_VANTAGE_API_KEY)
    result = analyze_stock(symbol, client)

    if result:
        if not initialize_output_file():
            print(f"[ERROR] Failed to initialize output file for {symbol}. Check permissions and path.")
            # Potentially return an error or raise an exception if file init is critical
        else:
            append_stock_result(result)
            print(f"Analysis for {symbol} completed and saved to {OUTPUT_FILE}")

        if 'predictions' in result and result['predictions'] is not None: # Ensure predictions exist
            pred = result['predictions']
            print(f"\nPrice Predictions for {symbol}:")
            print(f"30-Day Target: ${pred.get('price_target_30d', 0):.2f} ({pred.get('expected_return_30d', 0):.2f}%)")
            print(f"60-Day Target: ${pred.get('price_target_60d', 0):.2f} ({pred.get('expected_return_60d', 0):.2f}%)")

            if 'plot_path' in result and result['plot_path'] is not None:
                print(f"Prediction Plot for {symbol} saved to: {result['plot_path']}")
    else:
        print(f"Analysis for {symbol} failed. See log for details.")
    
    return result

# Main function
def main():
    """Main function to run the stock analysis"""
    print("\n===== ENHANCED STOCK ANALYZER =====")
    print("Using advanced ML and mean reversion analysis with price prediction")
    print("=" * 50 + "\n")

    try:
        # Check if command line arguments were provided
        import sys
        if len(sys.argv) > 1:
            # Use the first argument as the stock symbol
            symbol = sys.argv[1].strip().upper()
            print(f"[INFO] Using command line argument for symbol: {symbol}")
            perform_analysis_for_server(symbol) # user_id is optional
            return # Exit after command-line analysis

        # Interactive mode if no command line arguments
        # Alpha Vantage client for symbol search in interactive mode
        search_client = AlphaVantageClient(ALPHA_VANTAGE_API_KEY)
        while True:
            print("\nOptions:")
            print("1. Analyze a stock")
            print("2. Search for a stock")
            print("3. Exit")

            choice = input("Select an option (1-3): ").strip()

            if choice == '1':
                symbol = input("Enter stock symbol to analyze: ").strip().upper()
                if not symbol:
                    print("Please enter a valid stock symbol.")
                    continue
                perform_analysis_for_server(symbol) # user_id is optional

            elif choice == '2':
                keywords = input("Enter company name or keywords to search: ").strip()
                if not keywords:
                    print("Please enter valid search terms.")
                    continue
                
                matches = search_client.get_symbol_search(keywords) # Use search_client here

                if matches:
                    print("\nMatching stocks:")
                    print(f"{'Symbol':<10} {'Type':<8} {'Region':<8} Name")
                    print("-" * 70)
                    for match in matches:
                        print(f"{match['symbol']:<10} {match['type']:<8} {match['region']:<8} {match['name']}")
                    
                    analyze_choice = input("\nWould you like to analyze one of these stocks? (y/n): ").strip().lower()
                    if analyze_choice == 'y':
                        symbol_to_analyze = input("Enter the symbol to analyze: ").strip().upper()
                        if symbol_to_analyze:
                            perform_analysis_for_server(symbol_to_analyze) # user_id is optional
                else:
                    print("No matching stocks found.")

            elif choice == '3':
                print("Exiting program. Thank you!")
                break

            else:
                print("Invalid option. Please select 1, 2, or 3.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in the main function: {e}")
        traceback.print_exc()
    finally:
        print("[INFO] Analysis complete.")


if __name__ == "__main__":
    main()
