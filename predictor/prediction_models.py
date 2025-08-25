import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from datetime import datetime, timedelta
import random

class LotteryPredictor:
    def __init__(self, draws_data, config=None):
        """
        Initialize the predictor with historical lottery data
        
        Args:
            draws_data: List of LotteryDraw objects
            config: Dictionary with configuration parameters
        """
        self.draws = draws_data
        self.config = config or {}
        self.recency_weight = self.config.get('recency_weight', 0.7)
        self.lookback_periods = self.config.get('lookback_periods', 50)
        
        # Convert to DataFrame for easier manipulation
        self.df = self._prepare_dataframe()
        
    def _prepare_dataframe(self):
        """Prepare dataframe from lottery draws"""
        # Convert QuerySet to DataFrame
        data = []
        for draw in self.draws:
            data.append({
                'draw_date': draw.draw_date,
                'draw_number': draw.draw_number,
                'number_1': draw.number_1,
                'number_2': draw.number_2,
                'number_3': draw.number_3,
                'number_4': draw.number_4,
                'number_5': draw.number_5,
                'powerball': draw.powerball,
            })
        
        df = pd.DataFrame(data)
        
        # Ensure draw_date is datetime type
        df['draw_date'] = pd.to_datetime(df['draw_date'])
        
        # Extract date features
        df['day_of_week'] = df['draw_date'].dt.dayofweek
        df['month'] = df['draw_date'].dt.month
        df['day_of_year'] = df['draw_date'].dt.dayofyear
        
        # Calculate derived features
        # Sum of main numbers
        df['sum_main'] = df['number_1'] + df['number_2'] + df['number_3'] + df['number_4'] + df['number_5']
        
        # Mean of main numbers
        df['mean_main'] = df['sum_main'] / 5
        
        # Standard deviation of main numbers
        for idx, row in df.iterrows():
            numbers = [row['number_1'], row['number_2'], row['number_3'], row['number_4'], row['number_5']]
            df.at[idx, 'std_main'] = np.std(numbers)
        
        # Differences between consecutive numbers
        df['diff_1'] = df['number_2'] - df['number_1']
        df['diff_2'] = df['number_3'] - df['number_2']
        df['diff_3'] = df['number_4'] - df['number_3']
        df['diff_4'] = df['number_5'] - df['number_4']
        
        # Squared values for clustering
        for i in range(1, 6):
            df[f'number_{i}_squared'] = df[f'number_{i}'] ** 2
        
        # Apply lookback limit if specified
        if self.config.get('lookback_periods'):
            df = df.tail(self.config.get('lookback_periods'))
        
        return df
    
    def _random_prediction(self):
        """Generate a random prediction when other models can't be used"""
        # Generate 5 unique random numbers for main numbers
        main_numbers = sorted(random.sample(range(1, 51), 5))
        
        # Generate random powerball
        powerball = random.randint(1, 20)
        
        return {
            'number_1': main_numbers[0],
            'number_2': main_numbers[1],
            'number_3': main_numbers[2],
            'number_4': main_numbers[3],
            'number_5': main_numbers[4],
            'powerball': powerball,
            'confidence': 10.0  # Low confidence for random prediction
        }
    
    def frequency_model(self):
        """Predict based on frequency analysis with recency weighting"""
        if len(self.df) < 10:
            return self._random_prediction()
            
        # Use only recent draws based on lookback
        recent_df = self.df.tail(self.lookback_periods)
        
        # Calculate weights based on recency
        weights = np.linspace(1-self.recency_weight, 1, len(recent_df))
        
        # Count frequency for main numbers
        main_freq = {}
        for i in range(1, 51):
            main_freq[i] = 0
            
        for i in range(1, 6):
            col = f'number_{i}'
            for idx, num in enumerate(recent_df[col]):
                main_freq[num] += weights[idx]
                
        # Count frequency for powerball
        pb_freq = {}
        for i in range(1, 21):
            pb_freq[i] = 0
            
        for idx, num in enumerate(recent_df['powerball']):
            pb_freq[num] += weights[idx]
            
        # Sort by frequency and get top numbers
        main_numbers = sorted(main_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        main_numbers = sorted([x[0] for x in main_numbers])
        
        powerball = sorted(pb_freq.items(), key=lambda x: x[1], reverse=True)[0][0]
        
        # Calculate confidence based on frequency distribution
        total_main_freq = sum(main_freq.values())
        selected_main_freq = sum(main_freq[num] for num in main_numbers)
        main_confidence = selected_main_freq / total_main_freq if total_main_freq > 0 else 0
        
        total_pb_freq = sum(pb_freq.values())
        pb_confidence = pb_freq[powerball] / total_pb_freq if total_pb_freq > 0 else 0
        
        confidence = (main_confidence * 0.8) + (pb_confidence * 0.2)
        
        return {
            'number_1': main_numbers[0],
            'number_2': main_numbers[1],
            'number_3': main_numbers[2],
            'number_4': main_numbers[3],
            'number_5': main_numbers[4],
            'powerball': powerball,
            'confidence': min(confidence * 100, 99.9)
        }
    
    def pattern_model(self):
        """Predict based on pattern analysis"""
        if len(self.df) < 20:
            return self._random_prediction()
            
        recent_df = self.df.tail(self.lookback_periods)
        
        # Check if 'sum_main' exists in the dataframe
        if 'sum_main' not in recent_df.columns:
            return self._random_prediction()
        
        try:
            # Analyze sum patterns
            sum_counts = recent_df['sum_main'].value_counts().sort_index()
            
            # Check if we have enough data
            if len(sum_counts) < 2:
                return self._random_prediction()
                
            sum_range = pd.cut(sum_counts.index, bins=min(10, len(sum_counts)))
            
            # Check if sum_range is valid
            if sum_range is None or len(sum_range) == 0:
                return self._random_prediction()
                
            # Fix the FutureWarning by explicitly setting observed=False
            try:
                sum_range_counts = pd.Series(sum_counts.values, index=sum_range).groupby(level=0, observed=False).sum()
            except Exception as e:
                print(f"Error in groupby operation: {str(e)}")
                # Try without the observed parameter for older pandas versions
                try:
                    sum_range_counts = pd.Series(sum_counts.values, index=sum_range).groupby(level=0).sum()
                except:
                    return self._random_prediction()
            
            if len(sum_range_counts) == 0:
                return self._random_prediction()
                
            target_sum_range = sum_range_counts.idxmax()
            
            # Get min and max of the target sum range
            target_min = target_sum_range.left
            target_max = target_sum_range.right
            
            # Ensure target_min and target_max are valid numbers
            if target_min is None or target_max is None:
                return self._random_prediction()
                
        except Exception as e:
            # Handle any exceptions that occur during pattern analysis
            print(f"Error in pattern analysis: {str(e)}")
            return self._random_prediction()
        
        # Analyze odd/even patterns
        odd_counts = []
        for i in range(1, 6):
            odd_count = (recent_df[f'number_{i}'] % 2).sum()
            odd_counts.append(odd_count)
        
        avg_odd_count = sum(odd_counts) / len(odd_counts)
        target_odd_count = round(avg_odd_count)
        
        # Analyze high/low patterns (1-25 vs 26-50)
        high_counts = []
        for i in range(1, 6):
            high_count = (recent_df[f'number_{i}'] > 25).sum()
            high_counts.append(high_count)
        
        avg_high_count = sum(high_counts) / len(high_counts)
        target_high_count = round(avg_high_count)
        
        # Generate numbers that match the patterns
        main_numbers = []
        attempts = 0
        
        while len(main_numbers) < 5 and attempts < 1000:
            attempts += 1
            candidate_set = set()
            
            # Add numbers based on odd/even pattern
            odd_needed = target_odd_count
            high_needed = target_high_count
            
            while len(candidate_set) < 5:
                num = random.randint(1, 50)
                
                if num in candidate_set:
                    continue
                    
                is_odd = num % 2 == 1
                is_high = num > 25
                
                if is_odd and odd_needed > 0:
                    candidate_set.add(num)
                    odd_needed -= 1
                elif not is_odd and (5 - target_odd_count) > 0:
                    candidate_set.add(num)
                    target_odd_count += 1
                elif is_high and high_needed > 0:
                    candidate_set.add(num)
                    high_needed -= 1
                elif not is_high and (5 - target_high_count) > 0:
                    candidate_set.add(num)
                    target_high_count += 1
                else:
                    candidate_set.add(num)
            
            candidate_list = sorted(list(candidate_set))
            candidate_sum = sum(candidate_list)
            
            if target_min <= candidate_sum <= target_max:
                main_numbers = candidate_list
                break
        
        if not main_numbers:
            main_numbers = sorted(random.sample(range(1, 51), 5))
        
        # For powerball, analyze odd/even pattern
        pb_odd_count = (recent_df['powerball'] % 2).sum()
        pb_odd_ratio = pb_odd_count / len(recent_df)
        
        if random.random() < pb_odd_ratio:
            # Pick an odd powerball
            powerball = random.choice([i for i in range(1, 21) if i % 2 == 1])
        else:
            # Pick an even powerball
            powerball = random.choice([i for i in range(1, 21) if i % 2 == 0])
        
        # Calculate confidence based on pattern strength
        pattern_strength = 0.5  # Base confidence
        
        # Adjust based on how well we matched the patterns
        if target_min <= sum(main_numbers) <= target_max:
            pattern_strength += 0.2
            
        odd_count_actual = sum(1 for num in main_numbers if num % 2 == 1)
        if abs(odd_count_actual - target_odd_count) <= 1:
            pattern_strength += 0.15
            
        high_count_actual = sum(1 for num in main_numbers if num > 25)
        if abs(high_count_actual - target_high_count) <= 1:
            pattern_strength += 0.15
            
        return {
            'number_1': main_numbers[0],
            'number_2': main_numbers[1],
            'number_3': main_numbers[2],
            'number_4': main_numbers[3],
            'number_5': main_numbers[4],
            'powerball': powerball,
            'confidence': min(pattern_strength * 100, 99.9)
        }
    
    def clustering_model(self):
        """Predict based on clustering analysis"""
        if len(self.df) < 30:
            return self._random_prediction()
            
        try:
            # Prepare data for clustering
            features = []
            for i in range(1, 6):
                features.extend([f'number_{i}', f'number_{i}_squared'])
            
            features.extend(['sum_main', 'mean_main', 'std_main'])
            for i in range(1, 5):
                features.append(f'diff_{i}')
            
            # Check if all features exist in the dataframe
            missing_features = [f for f in features if f not in self.df.columns]
            if missing_features:
                return self._random_prediction()
                
            X = self.df[features].values
            
            # Normalize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal number of clusters
            inertia = []
            for k in range(2, min(11, len(X_scaled))):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)
                
            # Find elbow point or use default
            if len(inertia) > 3:
                diffs = np.diff(inertia)
                elbow = np.argmax(np.diff(diffs)) + 2
            else:
                elbow = 3
                
            # Cluster the data
            kmeans = KMeans(n_clusters=elbow, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Find the most recent cluster
            recent_cluster = clusters[-1]
            
            # Find all draws in the same cluster
            cluster_indices = np.where(clusters == recent_cluster)[0]
            
            # Get the draws from the same cluster
            cluster_draws = self.df.iloc[cluster_indices]
            
            # Calculate frequency within the cluster
            main_freq = {}
            for i in range(1, 51):
                main_freq[i] = 0
                
            for i in range(1, 6):
                col = f'number_{i}'
                for num in cluster_draws[col]:
                    main_freq[num] += 1
                    
            # Count frequency for powerball within the cluster
            pb_freq = {}
            for i in range(1, 21):
                pb_freq[i] = 0
                
            for num in cluster_draws['powerball']:
                pb_freq[num] += 1
                
            # Sort by frequency and get top numbers
            main_numbers = sorted(main_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            main_numbers = sorted([x[0] for x in main_numbers])
            
            powerball = sorted(pb_freq.items(), key=lambda x: x[1], reverse=True)[0][0]
            
            # Calculate confidence based on cluster quality
            cluster_size = len(cluster_indices)
            total_size = len(self.df)
            
            # Smaller clusters are more specific, so higher confidence
            cluster_confidence = 0.5 + (0.5 * (1 - (cluster_size / total_size)))
            
            return {
                'number_1': main_numbers[0],
                'number_2': main_numbers[1],
                'number_3': main_numbers[2],
                'number_4': main_numbers[3],
                'number_5': main_numbers[4],
                'powerball': powerball,
                'confidence': min(cluster_confidence * 100, 99.9)
            }
        except Exception as e:
            # Log the error if needed
            print(f"Error in clustering model: {str(e)}")
            return self._random_prediction()
    
    def regression_model(self):
        """Predict using regression analysis"""
        if len(self.df) < 40:
            return self._random_prediction()
            
        # Prepare features
        features = ['day_of_week', 'month', 'day_of_year']
        
        # Add lag features
        for lag in range(1, 6):
            for i in range(1, 6):
                col = f'number_{i}'
                self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
                features.append(f'{col}_lag_{lag}')
                
            self.df[f'powerball_lag_{lag}'] = self.df['powerball'].shift(lag)
            features.append(f'powerball_lag_{lag}')
            
        # Drop rows with NaN values
        df_reg = self.df.dropna()
        
        if len(df_reg) < 10:
            return self._random_prediction()
            
        # Prepare target variables
        targets = [f'number_{i}' for i in range(1, 6)] + ['powerball']
        
        # Train regression models for each number
        predictions = {}
        confidence_scores = []
        
        for target in targets:
            X = df_reg[features].values
            y = df_reg[target].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            
            # Calculate confidence
            score = model.score(X_test, y_test)
            confidence_scores.append(max(0.1, score))
            
            # Prepare prediction data
            last_row = self.df.iloc[-1:][features].values
            
            # Predict
            pred = model.predict(last_row)[0]
            
            # Round to nearest valid number
            if target == 'powerball':
                pred = max(1, min(20, round(pred)))
            else:
                pred = max(1, min(50, round(pred)))
                
            predictions[target] = pred
            
        # Ensure main numbers are unique
        main_numbers = [predictions[f'number_{i}'] for i in range(1, 6)]
        
        # If we have duplicates, replace them
        unique_nums = set()
        for i, num in enumerate(main_numbers):
            if num in unique_nums:
                # Find a number not in the set
                for new_num in range(1, 51):
                    if new_num not in unique_nums:
                        main_numbers[i] = new_num
                        break
            unique_nums.add(main_numbers[i])
            
        main_numbers.sort()
        
        # Calculate overall confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        return {
            'number_1': main_numbers[0],
            'number_2': main_numbers[1],
            'number_3': main_numbers[2],
            'number_4': main_numbers[3],
            'number_5': main_numbers[4],
            'powerball': predictions['powerball'],
            'confidence': min(avg_confidence * 100, 99.9)
        }
    
    def neural_network_model(self):
        """Predict using a neural network"""
        if len(self.df) < 100:
            return self._random_prediction()
            
        # Prepare features
        features = ['day_of_week', 'month', 'day_of_year']
        
        # Add lag features - reduce number of lags to improve performance
        for lag in range(1, 4):  # Reduced from 6 to 4
            for i in range(1, 6):
                col = f'number_{i}'
                self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
                features.append(f'{col}_lag_{lag}')
                
            self.df[f'powerball_lag_{lag}'] = self.df['powerball'].shift(lag)
            features.append(f'powerball_lag_{lag}')
            
        # Drop rows with NaN values
        df_nn = self.df.dropna()
        
        if len(df_nn) < 20:
            return self._random_prediction()
            
        # Prepare target variables
        targets = [f'number_{i}' for i in range(1, 6)] + ['powerball']
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(df_nn[features].values)
        
        # Train neural network models for each number
        predictions = {}
        confidence_scores = []
        
        for target in targets:
            try:
                y = df_nn[target].values
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model with simpler architecture and fewer iterations
                model = MLPRegressor(
                    hidden_layer_sizes=(25,),  # Simpler architecture
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    max_iter=500,  # Reduced iterations
                    early_stopping=True,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                
                # Calculate confidence
                score = model.score(X_test, y_test)
                confidence_scores.append(max(0.1, score))
                
                # Prepare prediction data
                last_row = scaler.transform(self.df.iloc[-1:][features].values)
                
                # Predict
                pred = model.predict(last_row)[0]
                
                # Round to nearest valid number
                if target == 'powerball':
                    pred = max(1, min(20, round(pred)))
                else:
                    pred = max(1, min(50, round(pred)))
                    
                predictions[target] = pred
            except Exception as e:
                print(f"Error in neural network for {target}: {str(e)}")
                # Fallback to a random number in the valid range
                if target == 'powerball':
                    predictions[target] = random.randint(1, 20)
                else:
                    predictions[target] = random.randint(1, 50)
                confidence_scores.append(0.1)  # Very low confidence for fallback
            
        # Ensure main numbers are unique
        main_numbers = [predictions[f'number_{i}'] for i in range(1, 6)]
        
        # If we have duplicates, replace them
        unique_nums = set()
        for i, num in enumerate(main_numbers):
            if num in unique_nums:
                # Find a number not in the set
                for new_num in range(1, 51):
                    if new_num not in unique_nums:
                        main_numbers[i] = new_num
                        break
            unique_nums.add(main_numbers[i])
            
        main_numbers.sort()
        
        # Calculate overall confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        return {
            'number_1': main_numbers[0],
            'number_2': main_numbers[1],
            'number_3': main_numbers[2],
            'number_4': main_numbers[3],
            'number_5': main_numbers[4],
            'powerball': predictions['powerball'],
            'confidence': min(avg_confidence * 100, 99.9)
        }
    
    def time_series_model(self):
        """Predict using time series analysis"""
        if len(self.df) < 50:
            return self._random_prediction()
            
        # We'll use a simplified approach for time series
        # For each number position, we'll create a time series and forecast
        
        predictions = {}
        confidence_scores = []
        
        for col in [f'number_{i}' for i in range(1, 6)] + ['powerball']:
            # Get the time series
            series = self.df[col].values
            
            try:
                # Use a simpler ARIMA model with fewer parameters
                model = ARIMA(series, order=(1, 0, 1))
                # Set a timeout for model fitting
                model_fit = model.fit(method='css', maxiter=50)
                
                # Forecast
                forecast = model_fit.forecast(steps=1)[0]
                
                # Round to nearest valid number
                if col == 'powerball':
                    pred = max(1, min(20, round(forecast)))
                else:
                    pred = max(1, min(50, round(forecast)))
                    
                predictions[col] = pred
                
                # Use AIC as a proxy for confidence
                aic = model_fit.aic
                # Convert AIC to a confidence score (lower AIC is better)
                conf = max(0.1, min(0.9, 1 - (aic / 1000)))
                confidence_scores.append(conf)
            except Exception as e:
                # If ARIMA fails, use a fallback approach
                # Use the average of recent values
                recent_values = self.df[col].tail(10).values
                pred = int(round(np.mean(recent_values)))
                
                # Ensure valid range
                if col == 'powerball':
                    pred = max(1, min(20, pred))
                else:
                    pred = max(1, min(50, pred))
                    
                predictions[col] = pred
                confidence_scores.append(0.3)  # Lower confidence for fallback
        
        # Ensure main numbers are unique
        main_numbers = [predictions[f'number_{i}'] for i in range(1, 6)]
        
        # If we have duplicates, replace them
        unique_nums = set()
        for i, num in enumerate(main_numbers):
            if num in unique_nums:
                # Find a number not in the set
                for new_num in range(1, 51):
                    if new_num not in unique_nums:
                        main_numbers[i] = new_num
                        break
            unique_nums.add(main_numbers[i])
            
        main_numbers.sort()
        
        # Calculate overall confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        return {
            'number_1': main_numbers[0],
            'number_2': main_numbers[1],
            'number_3': main_numbers[2],
            'number_4': main_numbers[3],
            'number_5': main_numbers[4],
            'powerball': predictions['powerball'],
            'confidence': min(avg_confidence * 100, 99.9)
        }
    
    def bayesian_model(self):
        """Predict using Bayesian analysis"""
        if len(self.df) < 30:
            return self._random_prediction()
            
        try:
            # Use only recent draws based on lookback
            recent_df = self.df.tail(self.lookback_periods)
            
            # Initialize prior probabilities (uniform)
            main_probs = {i: 1/50 for i in range(1, 51)}
            pb_probs = {i: 1/20 for i in range(1, 21)}
            
            # Update probabilities based on observed data
            for _, row in recent_df.iterrows():
                # Get the numbers from this draw
                main_numbers = [row[f'number_{i}'] for i in range(1, 6)]
                pb = row['powerball']
                
                # Update main number probabilities
                for num in range(1, 51):
                    # If the number appeared in this draw, increase its probability
                    if num in main_numbers:
                        main_probs[num] = main_probs[num] * 1.2
                    else:
                        main_probs[num] = main_probs[num] * 0.99
                
                # Update powerball probabilities
                for num in range(1, 21):
                    if num == pb:
                        pb_probs[num] = pb_probs[num] * 1.2
                    else:
                        pb_probs[num] = pb_probs[num] * 0.99
            
            # Normalize probabilities
            main_total = sum(main_probs.values())
            for num in main_probs:
                main_probs[num] /= main_total
                
            pb_total = sum(pb_probs.values())
            for num in pb_probs:
                pb_probs[num] /= pb_total
            
            # Select numbers with highest probabilities
            main_numbers = sorted(main_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            main_numbers = sorted([x[0] for x in main_numbers])
            
            powerball = sorted(pb_probs.items(), key=lambda x: x[1], reverse=True)[0][0]
            
            # Calculate confidence based on probability distribution
            # Higher concentration of probability mass = higher confidence
            top_main_probs = [main_probs[num] for num in main_numbers]
            main_confidence = sum(top_main_probs) / 5
            
            pb_confidence = pb_probs[powerball]
            
            # Combine confidences
            confidence = (main_confidence * 0.8) + (pb_confidence * 0.2)
            
            return {
                'number_1': main_numbers[0],
                'number_2': main_numbers[1],
                'number_3': main_numbers[2],
                'number_4': main_numbers[3],
                'number_5': main_numbers[4],
                'powerball': powerball,
                'confidence': min(confidence * 100, 99.9)
            }
        except Exception as e:
            print(f"Error in Bayesian model: {str(e)}")
            return self._random_prediction()