from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from datetime import datetime, timedelta

# Add these imports for charts
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

# Import your models and other modules
from .models import LotteryDraw, Prediction, PredictionAccuracy
from .forms import LotteryDrawForm, ImportDataForm, PredictionConfigForm
from .utils import import_powerball_data
from .prediction_models import LotteryPredictor

def index(request):
    """Home page with summary statistics and recent draws"""
    # Get recent draws
    recent_draws = LotteryDraw.objects.all().order_by('-draw_date')[:10]
    
    # Get recent predictions
    recent_predictions = Prediction.objects.all().order_by('-created_at')[:10]
    
    # Basic statistics
    total_draws = LotteryDraw.objects.count()
    
    # Frequency analysis for visualization
    number_frequency = {}
    powerball_frequency = {}
    
    # Initialize frequencies
    for i in range(1, 51):
        number_frequency[i] = 0
    for i in range(1, 21):
        powerball_frequency[i] = 0
    
    # Count frequencies from all draws
    for draw in LotteryDraw.objects.all():
        for i in range(1, 6):
            num = getattr(draw, f'number_{i}')
            number_frequency[num] += 1
        powerball_frequency[draw.powerball] += 1
    
    # Convert to lists for charts
    main_numbers = list(range(1, 51))
    main_freqs = [number_frequency[i] for i in main_numbers]
    
    pb_numbers = list(range(1, 21))
    pb_freqs = [powerball_frequency[i] for i in pb_numbers]
    
    # Create charts
    main_chart = create_bar_chart(main_numbers, main_freqs, 'Main Number Frequency', 'Number', 'Frequency')
    pb_chart = create_bar_chart(pb_numbers, pb_freqs, 'Powerball Frequency', 'Number', 'Frequency')
    
    context = {
        'recent_draws': recent_draws,
        'recent_predictions': recent_predictions,
        'total_draws': total_draws,
        'main_chart': main_chart,
        'pb_chart': pb_chart,
    }
    
    return render(request, 'predictor/index.html', context)

def create_bar_chart(x, y, title, xlabel, ylabel):
    """Helper function to create a bar chart"""
    plt.figure(figsize=(10, 6))
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.3)
    
    # Save to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    # Convert to base64 for embedding in HTML
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    
    return graphic

def draw_list(request):
    """View all lottery draws with pagination"""
    draws = LotteryDraw.objects.all().order_by('-draw_date')
    
    # Pagination
    paginator = Paginator(draws, 20)  # 20 draws per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
    }
    
    return render(request, 'predictor/draw_list.html', context)

def add_draw(request):
    """Add a new lottery draw"""
    if request.method == 'POST':
        form = LotteryDrawForm(request.POST)
        if form.is_valid():
            # Check for duplicate draw date
            draw_date = form.cleaned_data['draw_date']
            draw_number = form.cleaned_data['draw_number']
            
            if LotteryDraw.objects.filter(draw_date=draw_date).exists():
                messages.error(request, f"A draw for {draw_date} already exists.")
                return render(request, 'predictor/add_draw.html', {'form': form})
            
            # Validate that numbers are unique
            numbers = [
                form.cleaned_data['number_1'],
                form.cleaned_data['number_2'],
                form.cleaned_data['number_3'],
                form.cleaned_data['number_4'],
                form.cleaned_data['number_5'],
            ]
            
            if len(set(numbers)) != 5:
                messages.error(request, "The main numbers must be unique.")
                return render(request, 'predictor/add_draw.html', {'form': form})
            
            # Save the form
            form.save()
            messages.success(request, "Draw added successfully!")
            return redirect('draw_list')
    else:
        form = LotteryDrawForm()
    
    return render(request, 'predictor/add_draw.html', {'form': form})

def import_data(request):
    """Import lottery data from CSV or Excel file"""
    if request.method == 'POST':
        form = ImportDataForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            
            # Process the uploaded file
            success_count, error_count, error_messages = import_powerball_data(file)
            
            if success_count > 0:
                messages.success(request, f"Successfully imported {success_count} draws.")
            
            if error_count > 0:
                messages.warning(request, f"Encountered {error_count} errors during import.")
                for error in error_messages[:5]:  # Show first 5 errors
                    messages.error(request, error)
                
                if len(error_messages) > 5:
                    messages.error(request, f"... and {len(error_messages) - 5} more errors.")
            
            return redirect('draw_list')
    else:
        form = ImportDataForm()
    
    return render(request, 'predictor/import_data.html', {'form': form})

def generate_prediction(request):
    """Generate lottery prediction"""
    if request.method == 'POST':
        form = PredictionConfigForm(request.POST)
        if form.is_valid():
            model_type = form.cleaned_data['model_type']
            draw_date = form.cleaned_data['draw_date']
            recency_weight = form.cleaned_data['recency_weight']
            lookback_periods = form.cleaned_data['lookback_periods']
            
            # Get historical data
            draws = LotteryDraw.objects.all().order_by('draw_date')
            
            if draws.count() < 10:
                messages.error(request, "Not enough historical data for prediction. Please add at least 10 draws.")
                return render(request, 'predictor/generate_prediction.html', {'form': form})
            
            # Configure predictor
            config = {
                'recency_weight': recency_weight,
                'lookback_periods': lookback_periods,
            }
            
            predictor = LotteryPredictor(draws, config)
            
            # Generate prediction based on model type
            if model_type == 'frequency':
                result = predictor.frequency_model()
            elif model_type == 'pattern':
                result = predictor.pattern_model()
            elif model_type == 'clustering':
                result = predictor.clustering_model()
            elif model_type == 'regression':
                result = predictor.regression_model()
            elif model_type == 'neural':
                result = predictor.neural_network_model()
            elif model_type == 'time_series':
                result = predictor.time_series_model()
            elif model_type == 'bayesian':
                result = predictor.bayesian_model()
            elif model_type == 'ensemble':
                # Ensemble combines all models
                models = [
                    predictor.frequency_model(),
                    predictor.pattern_model(),
                    predictor.clustering_model(),
                    predictor.regression_model(),
                ]
                
                # If we have enough data, add more complex models
                if draws.count() >= 50:
                    models.append(predictor.time_series_model())
                if draws.count() >= 100:
                    models.append(predictor.neural_network_model())
                
                # Weighted voting based on confidence
                votes = {}
                for i in range(1, 51):
                    votes[i] = 0
                
                pb_votes = {}
                for i in range(1, 21):
                    pb_votes[i] = 0
                
                # Count votes weighted by confidence
                for model_result in models:
                    weight = model_result['confidence'] / 100
                    for i in range(1, 6):
                        num = model_result[f'number_{i}']
                        votes[num] += weight
                    
                    pb_votes[model_result['powerball']] += weight
                
                # Get top voted numbers
                main_numbers = sorted(votes.items(), key=lambda x: x[1], reverse=True)[:5]
                main_numbers = sorted([x[0] for x in main_numbers])
                
                powerball = sorted(pb_votes.items(), key=lambda x: x[1], reverse=True)[0][0]
                
                # Calculate ensemble confidence
                avg_confidence = sum(model['confidence'] for model in models) / len(models)
                
                result = {
                    'number_1': main_numbers[0],
                    'number_2': main_numbers[1],
                    'number_3': main_numbers[2],
                    'number_4': main_numbers[3],
                    'number_5': main_numbers[4],
                    'powerball': powerball,
                    'confidence': avg_confidence,
                }
            
            # Save prediction to database
            prediction = Prediction(
                draw_date=draw_date,
                model_type=model_type,
                number_1=result['number_1'],
                number_2=result['number_2'],
                number_3=result['number_3'],
                number_4=result['number_4'],
                number_5=result['number_5'],
                powerball=result['powerball'],
                confidence_score=result['confidence'],
            )
            prediction.save()
            
            messages.success(request, "Prediction generated successfully!")
            return redirect('prediction_detail', pk=prediction.pk)
    else:
        # Default to next Tuesday or Friday
        today = datetime.now().date()
        days_ahead = 1
        while days_ahead < 7:
            next_date = today + timedelta(days=days_ahead)
            if next_date.weekday() in [1, 4]:  # Tuesday (1) or Friday (4)
                break
            days_ahead += 1
        
        form = PredictionConfigForm(initial={'draw_date': next_date})
    
    return render(request, 'predictor/generate_prediction.html', {'form': form})

def prediction_list(request):
    """View all predictions with pagination"""
    predictions = Prediction.objects.all().order_by('-created_at')
    
    # Pagination
    paginator = Paginator(predictions, 20)  # 20 predictions per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
    }
    
    return render(request, 'predictor/prediction_list.html', context)

def prediction_detail(request, pk):
    """View details of a specific prediction"""
    prediction = get_object_or_404(Prediction, pk=pk)
    
    # Check if there's an actual draw for this date
    actual_draw = LotteryDraw.objects.filter(draw_date=prediction.draw_date).first()
    
    # Calculate accuracy if actual draw exists
    accuracy = None
    if actual_draw:
        # Count matching numbers
        pred_numbers = prediction.get_main_numbers()
        actual_numbers = actual_draw.get_main_numbers()
        
        matches = len(set(pred_numbers).intersection(set(actual_numbers)))
        powerball_match = prediction.powerball == actual_draw.powerball
        
        # Save accuracy
        accuracy, created = PredictionAccuracy.objects.get_or_create(
            prediction=prediction,
            actual_draw=actual_draw,
            defaults={
                'main_numbers_matched': matches,
                'powerball_matched': powerball_match,
            }
        )
        
        if not created:
            accuracy.main_numbers_matched = matches
            accuracy.powerball_matched = powerball_match
            accuracy.save()
    
    context = {
        'prediction': prediction,
        'actual_draw': actual_draw,
        'accuracy': accuracy,
    }
    
    return render(request, 'predictor/prediction_detail.html', context)

def statistics(request):
    """View detailed statistics and visualizations"""
    # Get all draws
    draws = LotteryDraw.objects.all().order_by('draw_date')
    
    if draws.count() < 5:
        messages.warning(request, "Not enough data for meaningful statistics. Please add more draws.")
        return redirect('index')
    
    # Prepare data for analysis
    draw_dates = [draw.draw_date for draw in draws]
    
    # Number frequency analysis
    number_frequency = {}
    powerball_frequency = {}
    
    # Initialize frequencies
    for i in range(1, 51):
        number_frequency[i] = 0
    for i in range(1, 21):
        powerball_frequency[i] = 0
    
    # Count frequencies from all draws
    for draw in draws:
        for i in range(1, 6):
            num = getattr(draw, f'number_{i}')
            number_frequency[num] += 1
        powerball_frequency[draw.powerball] += 1
    
    # Convert to lists for charts
    main_numbers = list(range(1, 51))
    main_freqs = [number_frequency[i] for i in main_numbers]
    
    pb_numbers = list(range(1, 21))
    pb_freqs = [powerball_frequency[i] for i in pb_numbers]
    
    # Create frequency charts
    main_chart = create_bar_chart(main_numbers, main_freqs, 'Main Number Frequency', 'Number', 'Frequency')
    pb_chart = create_bar_chart(pb_numbers, pb_freqs, 'Powerball Frequency', 'Number', 'Frequency')
    
    # Time-based analysis
    # Group by month
    month_data = {}
    for draw in draws:
        month = draw.draw_date.month
        if month not in month_data:
            month_data[month] = []
        month_data[month].extend(draw.get_main_numbers())
    
    # Calculate average for each month
    month_avgs = {}
    for month, numbers in month_data.items():
        month_avgs[month] = sum(numbers) / len(numbers)
    
    # Sort by month
    months = sorted(month_avgs.keys())
    month_avg_values = [month_avgs[m] for m in months]
    
    # Create month chart
    month_chart = create_bar_chart(months, month_avg_values, 'Average Number by Month', 'Month', 'Average Number')
    
    # Hot and cold numbers
    # Use last 10 draws for hot numbers
    recent_draws = draws.order_by('-draw_date')[:10]
    hot_numbers = {}
    
    for i in range(1, 51):
        hot_numbers[i] = 0
    
    for draw in recent_draws:
        for i in range(1, 6):
            num = getattr(draw, f'number_{i}')
            hot_numbers[num] += 1
    
    # Get top 10 hot numbers
    hot_nums = sorted(hot_numbers.items(), key=lambda x: x[1], reverse=True)[:10]
    hot_nums = [x[0] for x in hot_nums]
    hot_freqs = [hot_numbers[num] for num in hot_nums]
    
    # Create hot numbers chart
    hot_chart = create_bar_chart(hot_nums, hot_freqs, 'Hot Numbers (Last 10 Draws)', 'Number', 'Frequency')
    
    # Cold numbers (not drawn in last 20 draws)
    recent_20_draws = draws.order_by('-draw_date')[:20]
    recent_numbers = set()
    
    for draw in recent_20_draws:
        recent_numbers.update(draw.get_main_numbers())
    
    cold_nums = [num for num in range(1, 51) if num not in recent_numbers]
    
    # Prediction accuracy analysis
    accuracies = PredictionAccuracy.objects.all()
    
    model_accuracy = {}
    for model_type, _ in Prediction.MODEL_CHOICES:
        model_accuracy[model_type] = {
            'count': 0,
            'matches': [0, 0, 0, 0, 0, 0],  # 0, 1, 2, 3, 4, 5 matches
            'powerball': 0,
        }
    
    for acc in accuracies:
        model_type = acc.prediction.model_type
        model_accuracy[model_type]['count'] += 1
        model_accuracy[model_type]['matches'][acc.main_numbers_matched] += 1
        if acc.powerball_matched:
            model_accuracy[model_type]['powerball'] += 1
    
    # Calculate percentages
    for model_type in model_accuracy:
        if model_accuracy[model_type]['count'] > 0:
            for i in range(6):
                model_accuracy[model_type]['matches'][i] = (
                    model_accuracy[model_type]['matches'][i] / model_accuracy[model_type]['count'] * 100
                )
            model_accuracy[model_type]['powerball'] = (
                model_accuracy[model_type]['powerball'] / model_accuracy[model_type]['count'] * 100
            )
    
    context = {
        'total_draws': draws.count(),
        'main_chart': main_chart,
        'pb_chart': pb_chart,
        'month_chart': month_chart,
        'hot_chart': hot_chart,
        'hot_numbers': hot_nums,
        'cold_numbers': cold_nums,
        'model_accuracy': model_accuracy,
    }
    
    return render(request, 'predictor/statistics.html', context)

@require_POST
def refresh_prediction(request, pk):
    """Refresh a prediction with the latest data"""
    prediction = get_object_or_404(Prediction, pk=pk)
    
    # Get historical data
    draws = LotteryDraw.objects.all().order_by('draw_date')
    
    if draws.count() < 10:
        return JsonResponse({'error': 'Not enough historical data for prediction.'}, status=400)
    
    # Configure predictor
    config = {
        'recency_weight': 0.7,  # Default
        'lookback_periods': 50,  # Default
    }
    
    predictor = LotteryPredictor(draws, config)
    
    # Generate prediction based on model type
    model_type = prediction.model_type
    
    if model_type == 'frequency':
        result = predictor.frequency_model()
    elif model_type == 'pattern':
        result = predictor.pattern_model()
    elif model_type == 'clustering':
        result = predictor.clustering_model()
    elif model_type == 'regression':
        result = predictor.regression_model()
    elif model_type == 'neural':
        result = predictor.neural_network_model()
    elif model_type == 'time_series':
        result = predictor.time_series_model()
    elif model_type == 'bayesian':
        result = predictor.bayesian_model()
    elif model_type == 'ensemble':
        # Ensemble combines all models
        models = [
            predictor.frequency_model(),
            predictor.pattern_model(),
            predictor.clustering_model(),
            predictor.regression_model(),
        ]
        
        # If we have enough data, add more complex models
        if draws.count() >= 50:
            models.append(predictor.time_series_model())
        if draws.count() >= 100:
            models.append(predictor.neural_network_model())
        
        # Weighted voting based on confidence
        votes = {}
        for i in range(1, 51):
            votes[i] = 0
        
        pb_votes = {}
        for i in range(1, 21):
            pb_votes[i] = 0
        
        # Count votes weighted by confidence
        for model_result in models:
            weight = model_result['confidence'] / 100
            for i in range(1, 6):
                num = model_result[f'number_{i}']
                votes[num] += weight
            
            pb_votes[model_result['powerball']] += weight
        
        # Get top voted numbers
        main_numbers = sorted(votes.items(), key=lambda x: x[1], reverse=True)[:5]
        main_numbers = sorted([x[0] for x in main_numbers])
        
        powerball = sorted(pb_votes.items(), key=lambda x: x[1], reverse=True)[0][0]
        
        # Calculate ensemble confidence
        avg_confidence = sum(model['confidence'] for model in models) / len(models)
        
        result = {
            'number_1': main_numbers[0],
            'number_2': main_numbers[1],
            'number_3': main_numbers[2],
            'number_4': main_numbers[3],
            'number_5': main_numbers[4],
            'powerball': powerball,
            'confidence': avg_confidence,
        }
    
    # Update prediction
    prediction.number_1 = result['number_1']
    prediction.number_2 = result['number_2']
    prediction.number_3 = result['number_3']
    prediction.number_4 = result['number_4']
    prediction.number_5 = result['number_5']
    prediction.powerball = result['powerball']
    prediction.confidence_score = result['confidence']
    prediction.save()
    
    return JsonResponse({
        'success': True,
        'numbers': [
            prediction.number_1,
            prediction.number_2,
            prediction.number_3,
            prediction.number_4,
            prediction.number_5,
        ],
        'powerball': prediction.powerball,
        'confidence': prediction.confidence_score,
    })

# Make sure this function is in your views.py file
@require_POST
def clear_all_predictions(request):
    try:
        # Delete all predictions
        Prediction.objects.all().delete()
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
