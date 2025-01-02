# local files
from config import X, y, frame

# Statistical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Shiny App
from shiny import reactive
from shiny.express import render, ui
from shinywidgets import render_plotly

# SkLearn Library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ui.tags.style(
    """
    
    * {
        font-family: Arial;
    }
    
    body {
        background-color: #F7FAFC;
    }
    
    .nav {
        display: flex;                                  
        justify-content: center;                        /* Center items horizontally */
        align-items: center;                            /* Align items vertically */
        gap: 20px;                                      /* Space between nav items */
        background-color: #f8f9fa;                      /* Light grey background */
        padding: 10px 20px;                             /* Padding for the navbar */
        border-radius: 8px;                             /* Rounded corners */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);       /* Subtle shadow for depth */
    }
    
    /* Style for individual nav links */
    .nav-pills .nav-link {
        text-decoration: none;          
        color: #084081;                 
        font-size: 17px;                
        font-weight: bold;             
        padding: 10px 15px;             /* Add spacing around each link */
        border-radius: 5px;             /* Rounded corners */
        transition: all 1.0s ease;      /* Smooth hover effect */
    }

    /* Hover effect for nav links */
    .nav-pills .nav-link:hover {
        color: #084081;                
    }

    /* Active link style */
    .nav-pills .nav-link.active {
        color: #084081;
        background-color: transparent;
        text-decoration: underline;                    
    }

    
    h1 {
        color: #1C435A;
        text-align: center;
        margin-top: 20px;
        font-weight: 700;`
    }
    
    h3 {
        color: #1C435A;
        font-weight: 600px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    .card {
        color: #1C435A;
        border-color: #37A6A5;
    }
    
    .card-header {
        color: #1C435A;
        border-color: #37A6A5;
    }
    
    .modebar {
        display: none;
    }
    
    """
)

ui.h1("Boston Housing Machine Learning Model")

with ui.navset_pill(id='tab', selected='Linear Regression'):
    
    with ui.nav_panel(title="Model Summary"):
        
        "Panel A Content"
        
    with ui.nav_panel("Linear Regression"):
        
        # Div for Linear Regression Model
        with ui.div():
            
            ui.h3("Linear Regression Model")

            # Column wrap for our sample dataframe and linear regression model results
            with ui.layout_column_wrap(width=1/2):
                
                # Card for our sample dataframe
                with ui.card():
                    
                    ui.card_header("Dataset Sample")
                    
                    @render.data_frame
                    def frame_sample():
                        df = frame
                        
                        return df.head(5)
                        
                # Card for the linear regression model results    
                with ui.card():
                    
                    ui.card_header("Linear Regression Model Results")
                    
                    @render.data_frame
                    def linear_regression_results():
                        
                        lr, predict, r2, mean_abs_err, mean_sq_err = linear_regression_model()
                        
                        print(type(r2))
                        print(type(mean_abs_err))
                        
                        results_frame = pd.DataFrame(data={
                            'R2 Score': [round(r2, 4)],
                            'Mean Absolute Error': [round(mean_abs_err, 4)],
                            'Mean Squred Error': [round(mean_sq_err, 4)]
                        }).T.reset_index()
                        
                        results_frame.columns = ["Metric", "Value"]
                        
                        return results_frame
                        
            # Column wrap for Residual Plot and Predicted vs Actual Plot
            with ui.layout_column_wrap(width=1/2):
                
                # Residual Plot 
                with ui.card():
                    
                    ui.card_header('Residual Plot for Linear Regression Model')
                    
                    @render_plotly
                    def linear_regression_residual_plot():
                        
                        # Grab the necessary variables
                        X_test, X_train, y_test, y_train = create_testing_training_data()
                        lr, predict, r2, mean_abs_err, mean_sq_err = linear_regression_model() 
                        
                        # convert y_test to a numpy array
                        y_test = np.array(y_test)
                        
                        # Calculate the residuals and flatten/ravel into 1d arrays for dataframe creation
                        residuals = (predict - y_test).ravel()
                        predict = predict.flatten()
                        
                        print(predict.shape)
                        print(residuals.shape)
                        
                        # Create a dataframe for easier plotting
                        df = pd.DataFrame(
                            {
                                "Predicted": predict,
                                "Residuals": residuals
                            }
                        )
                        
                        # plot the results
                        fig = px.scatter(data_frame=df,
                                        x='Predicted',
                                        y='Residuals',
                                        color='Residuals',
                                        color_continuous_scale=px.colors.diverging.delta,
                                        title='Residual Plot For Linear Regression Model',
                                        labels={
                                            'Predicted': 'Predicted Values',
                                            'Residuals': 'Residual Values'
                                        })
                        
                        fig.add_hline(
                            y=0,
                            line_dash='dash',
                            line_color='#808080'
                        )
                        
                        fig.update_layout(
                            title=dict(x=0.5),
                            plot_bgcolor='#FCFCFC',
                            xaxis=dict(showgrid=True, gridcolor='#F0F7E8'),
                            yaxis=dict(showgrid=True, gridcolor='#F0F7E8'),
                            coloraxis_showscale=False
                            # font=dict(
                            #     size=12,
                            #     color='black' 
                            # ),
                        )
                        
                        return fig
                
                # Predicted vs Actual plot        
                with ui.card():
                
                    ui.card_header('Actual vs Predicted Values')
                    
                    @render_plotly
                    def linear_regression_predict_vs_actual():
                        
                        # Import our training and testing variables
                        X_test, X_train, y_test, y_train = create_testing_training_data()
                        
                        # import the results
                        lr, predict, r2, mean_abs_err, mean_sq_err = linear_regression_model()
                        
                        # create a dataframe for easier plotting 
                        df = pd.DataFrame(data={
                            "y_test": np.array(y_test).flatten(),
                            'prediction': predict.flatten()
                        })
                        
                        # Create the scatter plot
                        figure = px.scatter(data_frame=df,
                                            x='y_test',
                                            y='prediction',
                                            color='prediction',
                                            color_continuous_scale=px.colors.sequential.GnBu,
                                            title='Actual vs. Predicted Values',
                                            labels={
                                                'y_test': 'Actual Values',
                                                'prediction': 'Predicted Values'
                                            })
                        
                        # Create the diagonal line
                        figure.add_scatter(
                            x=[df['y_test'].min() - 2, df['y_test'].max()+ 2],
                            y=[df['y_test'].min() - 2, df['y_test'].max() + 2],
                            mode='lines',
                            line=dict(color='#808080',
                                    dash='dash',
                                    width=2),
                            name='Ideal Fit Line'
                        )
                        
                        figure.update_layout(
                            title=dict(x=0.5),
                            plot_bgcolor='#FCFCFC',
                            xaxis=dict(showgrid=True, gridcolor='#F0F7E8'),
                            yaxis=dict(showgrid=True, gridcolor='#F0F7E8'),
                            coloraxis_showscale=False
                            # font=dict(
                            #     size=12,
                            #     color='black' 
                            # ),
                        )
                        
                        return figure
                
            # Column Wrap for Features Importance
            with ui.layout_column_wrap(width=1/1):
            
                # Features Importance    
                with ui.card():
            
                    ui.card_header('Features Importance (Coefficients)')
                    
                    @render_plotly
                    def linear_regression_feature_importance():
                        
                        # Import our training and testing variables
                        X_test, X_train, y_test, y_train = create_testing_training_data()
                        
                        # import the results
                        lr, predict, r2, mean_abs_err, mean_sq_err = linear_regression_model()
                        
                        # Extract feature names and coefficients
                        features = frame.columns[:-1]
                        coefficients = lr.coef_.flatten()
                        
                        # create the plot
                        figure = px.bar(x=coefficients,
                                        y=features,
                                        orientation='h',
                                        color=coefficients,
                                        color_continuous_scale=px.colors.diverging.Tealrose,
                                        title='Features Importance',
                                        labels={
                                            'x': 'Coefficients',
                                            'y': 'Features'
                                        }
                                        )
                        
                        figure.update_layout(
                            title=dict(x=0.5),
                            plot_bgcolor='#FCFCFC',
                            xaxis=dict(showgrid=True, gridcolor='#F0F7E8'),
                            yaxis=dict(showgrid=True, gridcolor='#F0F7E8'),
                            coloraxis_showscale=False
                        )
                        
                        
                        return figure
                        
    with ui.nav_panel("Logistic Regression"):
        
        "TODO - CONTENT GOES HERE"          

# Reactive Calcs for Our Linear Regression Model
@reactive.Calc
def create_testing_training_data():
     """
     Create our Training & Testing data with SkLearn train_test_split function
     """
     X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.22, random_state=2792)
     
     return X_test, X_train, y_test, y_train
 
@reactive.calc    
def linear_regression_model():
    
    # Import our training and testing variables
    X_test, X_train, y_test, y_train = create_testing_training_data()
    
    """
    Create the model
    """
    lr = LinearRegression()
    
    """
    Train the model
    """
    lr.fit(X=X_train, y=y_train)
    
    """
    Make a prediction
    """
    predict = lr.predict(X_test)
    
    """
    Measure Accuracy
    """
    r2 = r2_score(y_true=y_test, y_pred=predict)
    mean_abs_err = mean_absolute_error(y_true=y_test, y_pred=predict)
    mean_sq_err = mean_squared_error(y_true=y_test, y_pred=predict)
    
    return lr, predict, r2, mean_abs_err, mean_sq_err

