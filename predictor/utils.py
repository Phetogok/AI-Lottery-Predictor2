import pandas as pd
from datetime import datetime
from .models import LotteryDraw
import numpy as np

def import_powerball_data(file_path):
    """
    Import South African Powerball data from Excel file with the specific format:
    Date, Day Of Week, 1, 2, 3, 4, 5, PowerBall
    """
    try:
        # Try different approaches to read the Excel file
        try:
            # First try with openpyxl engine
            df = pd.read_excel(file_path, skiprows=3, engine='openpyxl')
        except Exception as e1:
            try:
                # Then try with xlrd engine
                df = pd.read_excel(file_path, skiprows=3, engine='xlrd')
            except Exception as e2:
                try:
                    # Try reading as CSV
                    df = pd.read_csv(file_path, skiprows=3)
                except Exception as e3:
                    return 0, 1, [f"Failed to read file: tried openpyxl, xlrd, and csv formats. Error: {str(e1)}, {str(e2)}, {str(e3)}"]
        
        # Check if we have enough columns
        if len(df.columns) < 7:
            return 0, 1, ["File does not have enough columns. Expected at least 7 columns."]
        
        # Rename columns to match our expected format
        df.columns = ['date', 'day_of_week', 'number_1', 'number_2', 'number_3', 
                      'number_4', 'number_5', 'powerball'] + list(df.columns[8:])
        
        # Skip rows that have header-like values
        df = df[~df['date'].astype(str).str.contains('Date|date', na=False)]
        
        # Drop any rows with NaN values in essential columns
        df = df.dropna(subset=['date', 'number_1', 'powerball'])
        
        # Initialize counters
        success_count = 0
        error_count = 0
        error_messages = []
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Convert date to datetime object
                if isinstance(row['date'], str):
                    try:
                        # Try different date formats
                        try:
                            date_obj = datetime.strptime(row['date'], '%Y-%m-%d').date()
                        except:
                            try:
                                date_obj = datetime.strptime(row['date'], '%d/%m/%Y').date()
                            except:
                                date_obj = datetime.strptime(row['date'], '%d-%m-%Y').date()
                    except Exception as e:
                        error_count += 1
                        error_messages.append(f"Error parsing date '{row['date']}' at row {idx+4}: {str(e)}")
                        continue
                else:
                    date_obj = row['date']
                
                # Convert numbers to integers
                try:
                    number_1 = int(float(row['number_1']))
                    number_2 = int(float(row['number_2']))
                    number_3 = int(float(row['number_3']))
                    number_4 = int(float(row['number_4']))
                    number_5 = int(float(row['number_5']))
                    powerball = int(float(row['powerball']))
                except Exception as e:
                    error_count += 1
                    error_messages.append(f"Error converting numbers at row {idx+4}: {str(e)}")
                    continue
                
                # Validate number ranges
                if not (1 <= number_1 <= 50 and 1 <= number_2 <= 50 and 1 <= number_3 <= 50 and 
                        1 <= number_4 <= 50 and 1 <= number_5 <= 50 and 1 <= powerball <= 20):
                    error_count += 1
                    error_messages.append(f"Invalid number range at row {idx+4}")
                    continue
                
                # Check if this draw already exists
                existing_draw = LotteryDraw.objects.filter(draw_date=date_obj).first()
                
                if existing_draw:
                    # Update existing draw
                    existing_draw.number_1 = number_1
                    existing_draw.number_2 = number_2
                    existing_draw.number_3 = number_3
                    existing_draw.number_4 = number_4
                    existing_draw.number_5 = number_5
                    existing_draw.powerball = powerball
                    existing_draw.save()
                else:
                    # Create new draw
                    LotteryDraw.objects.create(
                        draw_date=date_obj,
                        number_1=number_1,
                        number_2=number_2,
                        number_3=number_3,
                        number_4=number_4,
                        number_5=number_5,
                        powerball=powerball
                    )
                
                success_count += 1
            
            except Exception as e:
                error_count += 1
                error_messages.append(f"Error processing row {idx+4}: {str(e)}")
        
        return success_count, error_count, error_messages
    
    except Exception as e:
        return 0, 1, [f"Error reading file: {str(e)}"]