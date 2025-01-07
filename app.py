# local files
from config import X, y, frame

# Statistical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Shiny App
from htmltools import HTML
from shiny import reactive
from shiny.express import render, ui, input
from shinywidgets import render_plotly

# SkLearn Library
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error, median_absolute_error

ui.tags.style(
    """
    
    * {
        font-family: Arial;
    }
    
    body {
        background-color: #F7FAFC;
    }
    
    br {
        line-height: 10px;
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
    
    h6 {
        color: black;
        text-decoration: underline;
        font-weight: 900px;
        margin-bottom: -10px;
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

with ui.navset_pill(id='tab', selected='Random Forests'):
    
    # PANEL 1 - Model Summaries
    with ui.nav_panel(title="Model Summary"):
        
        # column wrap summarizing the scores of each model
        with ui.layout_column_wrap():
            
            pass
            
    # PANEL 2 - Linear Regression Panel    
    with ui.nav_panel("Linear Regression"):
        
        # Div for Linear Regression Model
        with ui.div():
            
            ui.h3("Linear Regression Model")
            
            # Column wrap for model paramater selections
            with ui.layout_column_wrap(width=1/3):
                
                # Select Test Size
                ui.input_numeric(id="lr_test_size", label="Enter Size of Test Data: ",value=21, min=11, max=51, step=2)    
                
                # Apply a regularization method
                ui.input_select(  
                    id="lr_reg_selec",  
                    label="Select A Regularization Method: ",  
                    choices={"none": "No Method", "ridge": "Ridge Regression", "lasso": "Lasso Regression", 'elastic': "ElasticNet"},
                    selected='none',  
                )
            
            # Column wrap for our sample dataframe and linear regression model results
            with ui.layout_column_wrap(width=1/2):
                
                # Card for our sample dataframe
                with ui.card():
                    
                    ui.card_header("Dataset Sample")
                    
                    @render.data_frame
                    def frame_sample():
                        df = frame
                        
                        return df.head(10)
                        
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
                            'Mean Squared Error': [round(mean_sq_err, 4)]
                        }).T.reset_index()
                        
                        results_frame.columns = ["Metric", "Value"]
                        
                        results_frame['Description'] = [
                            "How well the regression line approximates the actual data.",
                            "Average of the absolute differences between the predicted and actual target values.",
                            "Average squared difference between the predicted and actual target values."
                        ]
                        
                        return results_frame
                        
            # Column wrap for Residual Plot and Predicted vs Actual Plot
            with ui.layout_column_wrap(width=1/2):
                
                # Residual Plot 
                with ui.card():
                    
                    with ui.layout_sidebar():
                        
                        with ui.sidebar(open='open', position='left'):
                            
                            ui.h6("Residual Plot Notes")
                            ui.HTML(
                                """
                                A Residual is the difference between the predicted value and the actual value. A small residual value and/or one close to the horizontal line represents a strong prediction. Siginifcant residual values indicate model improvements are needed.
                                """
                                
                            )
                        
                        ui.card_header('Residual Plot for Linear Regression Model')
                        
                        @render_plotly
                        def linear_regression_residual_plot():
                                
                                # Grab the necessary variables
                                X_train, X_test, y_train, y_test = create_testing_training_data()
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
                
                    with ui.layout_sidebar():
                        
                        with ui.sidebar(open='open', position='left'):
                            
                            ui.h6("Plot Notes")
                            ui.HTML(
                                """
                                This chart compares the predicted values to the actual values. Data points closest to the diagonal line represent strong/accurate predictions, whereas points far from the line represent areas of improvement within our model.
                                """
                            )
                            
                        ui.card_header('Actual vs Predicted Values')
                    
                        @render_plotly
                        def linear_regression_predict_vs_actual():
                            
                            # Import our training and testing variables
                            X_train, X_test, y_train, y_test = create_testing_training_data()
                            
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
                        X_train, X_test, y_train, y_test = create_testing_training_data()
                        
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
                    
                    ui.h6("What is Features Importance?")
                    ui.HTML(
                        """
                        <p style="font-size: 16px; line-height: 1.25; text-align: justify; color: #333;">
                            <strong>Features Importance</strong> refers to the relative importance of each feature 
                            (input data / independent variable) in contributing to the prediction of a machine learning model. 
                            It provides a quick and easily interpretable visual to determine which features have the highest 
                            and lowest impact on the model. <br><br>
                            
                            In <em>linear regression</em>, features importance is derived from the <strong>coefficients</strong>. 
                            The coefficients indicate how much the target variable changes for a unit increase in the feature, 
                            holding all the other features constant. The larger the coefficient, the higher the impact / importance 
                            on the end prediction. <br><br>
                            
                            Coefficients can have a <strong>negative</strong> or <strong>positive</strong> relationship with the 
                            target variable (AKA the <strong>Label</strong>). <br><br>
                            
                            <em>Note:</em> The coefficients above have been standardized.
                        </p>
                        """
                    )
    
    # PANEL - 3 K-Nearest Neighbours Model                                    
    with ui.nav_panel("K-Nearest Neighbours"):
        
        # Div for k-nearest neighbours
        with ui.div():
            
            # header
            ui.h3("KNN Model")
            
            # column wrap for neighbour selection
            with ui.layout_column_wrap():
                
                ui.input_numeric(id="knn_neigh", label="Select Number of Neighbours: ", value=3, min=3, max=101, step=2) 
            
            # Column wrap for sample dataset & model scores
            with ui.layout_column_wrap(width=(1/2)):
                
                # Sample dataframe for dataset
                with ui.card():
                    
                    ui.card_header("Dataset Sample")
                    
                    @render.data_frame
                    def frame_sample_2():
                        df = frame
                        
                        return df.head(10)
                
                # KNN Model Scores
                with ui.card():
                    
                    ui.card_header("KNN Model Scores ")
                    
                    @render.data_frame
                    def display_knn_scores():  
                        
                        prediction, r2, rmsq, median_abs_err =  knn_model()   
                        
                        results_frame = pd.DataFrame(data={
                           "R2 Score": [round(r2, 4)],
                           "Root Mean Squared Error": [round(rmsq, 4)],
                           "Median Absolute Error": [round(median_abs_err, 4)] 
                        }).T.reset_index()
                        
                        results_frame.columns = ['Metrics', 'Values']
                        results_frame['Description'] = 'TODO'
                        
                        return results_frame

            # Column wrap for Residual Plot & Actual vs Predicted Plot
            with ui.layout_column_wrap(width=1/2):
                
                # Residual Plot
                with ui.card():
                    
                    ui.card_header("Residual Plot")
                    
                    @render_plotly
                    def display_knn_residuals():
                        
                        
                        # Import the variabls to construct the residual plot
                        X_train, X_test, y_train, y_test = knn_create_testing_training_data()
                        prediction, r2, rmsq, median_abs_err = knn_model()
                        
                        # convert the y_test variable from a frame to a numpy array
                        y_test = np.array(y_test)
                        
                        # Calculate residuals = predicted values - actuals. flatten the results into a 1d array
                        residuals = (prediction - y_test).ravel()
                        prediction = prediction.flatten()
                        
                        print(f"Prediction Type: {type(prediction)}")
                        print(f"Prediction Shape: {prediction.shape}")
                        
                        print(f"Actual Value Type: {type(y_test)}")
                        print(f"Actual Value Shape: {y_test.shape}")
                        
                        # Create the residuals dataframe
                        residuals_frame = pd.DataFrame(data={
                            'Predict': prediction,
                            'Residuals': residuals
                        })
                        
                        # plot the scatter chart
                        figure = px.scatter(data_frame=residuals_frame,
                                            x='Predict',
                                            y='Residuals',
                                            title='Residual Plot for KNN Model',
                                            color='Residuals',
                                            color_continuous_scale=px.colors.diverging.delta
                                            )
                        
                        # add the horizontal line
                        figure.add_hline(
                                    y=0,
                                    line_dash='dash',
                                    line_color='#808080'
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
                        
                # Actual vs Predicted Plot
                with ui.card():
                
                    ui.card_header("Actual vs. Predicted Plot")
                    
                    @render_plotly
                    def display_knn_actual_predicted():
                        
                        # Import the variabls to construct the residual plot
                        X_train, X_test, y_train, y_test = knn_create_testing_training_data()
                        prediction, r2, rmsq, median_abs_err = knn_model()
                        
                        # Convert the frame to numpy array and flatten all variables. Dataframe takes a numpy array as arg
                        y_test = np.array(y_test).flatten()
                        prediction = prediction.flatten()
                        
                        # Create frame
                        frame = pd.DataFrame(data={
                            'Actual': y_test,
                            'Predicted': prediction,
                        })
                        
                        # Create the initial scatter figure
                        figure = px.scatter(data_frame=frame,
                                            x='Actual',
                                            y='Predicted',
                                            title='Actual vs. Predicted Values',
                                            labels={
                                                'Actual': 'Actual Values',
                                                'Predicted': 'Predicted Values'
                                            }, 
                                            color='Predicted',
                                            color_continuous_scale=px.colors.sequential.GnBu)
                        
                        
                        # Add the horizontal line
                        figure.add_scatter(
                            x=[frame['Actual'].min(), frame['Actual'].max()],
                            y=[frame['Predicted'].min(), frame['Predicted'].max()],
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
            
            # Column wrap for knn validation curve
            # TODO: KNN VALIDATION CURVE
    
    # PANEL - 4 Decision Tree                
    with ui.nav_panel("Regressive Decision Tree"):
        
        # div for decision trees
        with ui.div():
            
            ui.h3("Regression Decision Trees")
            
            # column wrap for hyperparameter tuning
            with ui.layout_column_wrap(width=1/4):
                
                # model test size
                ui.input_numeric(id="reg_tree_test_size", label="Select Test Size: ", value=37, min=10, max=40, step=1) 
                
                # select criterion
                ui.input_select(  
                    id="reg_tree_criterion",  
                    label="Select a Splitting Criterion: ",  
                    choices={"squared_error": "Squared Error",
                             "friedman_mse": "Friedman Mean Squared Error",
                             "absolute_error": "Absolute Error",
                             "poisson": "Poisson"},  
                    selected='squared_error'
                )  
                
                # model max depth
                ui.input_numeric(id="reg_tree_depth", label="Set Tree Depth: ", value=5, min=2, max=20, step=1)
                
                # model mininum leaf samples
                ui.input_numeric(id="reg_tree_min_leaf_samples", label="Set Minimum Samples For Leaf Node: ", value=4, min=2, max=20, step=1)  

            # column wrap for dataframe summary and accuracy scores
            with ui.layout_column_wrap(width=1/2):
                
                # Sample dataframe for dataset
                with ui.card():
                    
                    ui.card_header("Dataset Sample")
                    
                    @render.data_frame
                    def frame_sample_3():
                        df = frame
                        
                        return df.head(10)
                
                # Accuracy scores    
                with ui.card():
                    
                    ui.card_header("Decision Tree Scores")
                    
                    @render.data_frame
                    def display_tree_scores():
                        
                        # get variables
                        X_train, X_test, y_train, y_test, reg_tree, prediction, r2, mse, mae = reg_tree_model()
                        
                        # make results frame 
                        frame = pd.DataFrame(data={
                            "R2 Score": [round(r2, 4)],
                            "Mean Squared Error": [round(mse, 4)],
                            "Mean Absolute Error": [round(mae, 4)]
                        }).transpose().reset_index()
                        
                        frame.columns = ['Metrics', 'Values']
                        frame['Description'] = 'TODO'
                        
                        return frame
                
            # column wrap for residual plots and actual vs predicted values
            with ui.layout_column_wrap(width=1/2):
                
                # Card for residual plot
                with ui.card():
                    
                    ui.card_header("Residuals Plot")
                    
                    @render_plotly
                    def reg_tree_residuals():
                        
                        # import our variables
                        X_train, X_test, y_train, y_test, reg_tree, prediction, r2, mse, mae = reg_tree_model()
                        
                        # convert the y_test into a numpy array and then flatten into 1 dimension
                        y_test = np.array(y_test).flatten()
                        
                        # calculate residuals (which is the difference between actual and predicted)
                        residuals = (y_test - prediction).flatten()
                        
                        # create frame
                        frame = pd.DataFrame(data={
                            "Residuals": residuals,
                            "Actual": y_test
                        })
                        
                        # plot frame
                        figure = px.scatter(data_frame=frame,
                                            x='Actual',
                                            y='Residuals',
                                            title='Residuals Plot',
                                            color='Residuals',
                                            color_continuous_scale=px.colors.diverging.delta)
                        
                        # add the horizontal line
                        figure.add_hline(
                                    y=0,
                                    line_dash='dash',
                                    line_color='#808080'
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
                
                # Card for actual vs predicted plot
                with ui.card():
                    
                    ui.card_header("Actual vs. Predicted Values")        
                    
                    @render_plotly
                    def reg_tree_actual_predicted():
                        
                        # get our variables
                        X_train, X_test, y_train, y_test, reg_tree, prediction, r2, mse, mae = reg_tree_model()
                        
                        # flatten all the values we need
                        y_test = np.array(y_test).flatten()
                        prediction = prediction.flatten()
                        
                        # create the frame
                        frame = pd.DataFrame(data={
                            'Actual': y_test,
                            'Predicted': prediction
                        })
                        
                        # create the plot
                        figure = px.scatter(data_frame=frame,
                                            x='Actual',
                                            y='Predicted',
                                            title='Actual vs Predicted Values',
                                            color='Predicted',
                                            color_continuous_scale=px.colors.sequential.GnBu)
                        
                        # add the vertical line
                        figure.add_scatter(
                            x=[frame['Actual'].min(), frame['Actual'].max()],
                            y=[frame['Predicted'].min(), frame['Predicted'].max()],
                            mode='lines',
                                line=dict(color='#808080',
                                        dash='dash',
                                        width=2),
                                name='Ideal Fit Line'
                        )
                        
                        # update plot background and remove color scale
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

            # column wrap for feature importance
            with ui.layout_column_wrap(width=1/1):
                
                with ui.card():
                    
                    ui.card_header("Decison Tree Feature Importance")
                
                    @render_plotly
                    def reg_tree_feature_importance():
                        
                        # get variables
                        X_train, X_test, y_train, y_test, reg_tree, prediction, r2, mse, mae = reg_tree_model()
                        
                        # get important features and features
                        imp_features = reg_tree.feature_importances_
                        features = frame.columns[:-1]
                        
                        # Create a dataframe so we can plot
                        feature_frame = pd.DataFrame(data={
                            'Feature': features,
                            'Importance': imp_features
                        })
                        
                        # Create the plot
                        figure = px.bar(data_frame=feature_frame,
                                        orientation='h',
                                        x='Importance',
                                        y='Feature',
                                        title='Decision Tree Feature Importance',
                                        color='Importance',
                                        color_continuous_scale=px.colors.diverging.Tealrose)
                        
                        
                        figure.update_layout(
                            title=dict(x=0.5),
                            plot_bgcolor='#FCFCFC',
                            xaxis=dict(showgrid=True, gridcolor='#F0F7E8'),
                            yaxis=dict(showgrid=True, gridcolor='#F0F7E8'),
                            coloraxis_showscale=False
                        )
                        
                        return figure
                        
    # PANEL - 5 Random Forests  
    with ui.nav_panel("Random Forests"):
        
        # div for random forest models
        with ui.div():
            
            # Column layout wrap for hyperparameter tuning
            with ui.layout_column_wrap(width=1/3):
                
                # tune test size
                ui.input_numeric(id="rf_testsize", label="Select a Test Size: ", value=30, min=10, max=40, step=2) 
                
                # tune number of estimators / decision trees
                ui.input_numeric(id="rf_num_trees", label="Set Number of Trees: ", value=500, min=25, max=500, step=25) 
                
                # tune tree depth
                ui.input_numeric(id="rf_tree_depth", label="Set Tree Depth: ", value=14, min=2, max=20) 
        
            # Column Layout wrap for Sample DataSet and Accuracy Scores
            with ui.layout_column_wrap(width=1/2):           
                
                # Sample dataframe
                with ui.card():
                    
                    ui.card_header("Sample Dataset")
                    
                    @render.data_frame
                    def display_frame_5():
                        
                        df = frame
                        
                        return df.head(10)
                        
                # Accuracy Scores
                with ui.card():
                    
                    ui.card_header("Random Forest Accuracy Scores")
                    
                    @render.data_frame
                    def rf_accuracy_scores():
                        
                        # get scores
                        rf, predict, oob_score, r2, mae, mse = rf_model()
                        
                        # create frame
                        results_frame = pd.DataFrame(data={
                            "OOB Score": [round(oob_score, 4)],
                            "R2 Score": [round(r2, 4)],
                            "Mean Absolute Error": [round(mae, 4)],
                            "Mean Squared Error": [round(mse, 4)]
                        }).transpose().reset_index()
                        
                        results_frame.columns = ['Metrics', 'Values']
                        
                        results_frame['Description'] = 'TODO'
                        
                        return results_frame
            
            # Column layout wrap for Residual and Actual vs Predicted plot
            with ui.layout_column_wrap(width=1/2):
                
                # card for residual plot
                with ui.card():
                    
                    ui.card_header("Residual Plot")     
                    
                    @render_plotly
                    def rf_residual_plot():
                        
                        # get variables       
                        X_train, X_test, y_train, y_test = rf_training_testing_data()
                        rf, predict, oob_score, r2, mae, mse = rf_model()
                        
                        # calculate residuals. Predicted minus actual
                        y_test = np.array(y_test).flatten()
                        predict = predict.flatten()
                        
                        residual = (predict - y_test)
                        
                        # create dataframe
                        resid_frame = pd.DataFrame(data={
                            "Residuals": residual,
                            "Actual": y_test
                        })
                        
                        # create scatter plot
                        figure = px.scatter(
                            data_frame=resid_frame,
                            x='Actual',
                            y='Residuals',
                            title="Random Forest Residual Plot",
                            color='Residuals',
                            color_continuous_scale=px.colors.diverging.delta
                            )
                        
                        # add the horizontal line
                        figure.add_hline(
                                    y=0,
                                    line_dash='dash',
                                    line_color='#808080'
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
                        
                        # return figure
                        return figure

                # card for actual vs predicted
                with ui.card():
                    
                    ui.card_header("Actual vs. Predicted Values")

                    @render_plotly
                    def rf_actual_vs_predicted():

                        # get our variables      
                        X_train, X_test, y_train, y_test = rf_training_testing_data()
                        rf, predict, oob_score, r2, mae, mse = rf_model()

                        # convert to 1d numpy arrays
                        y_test = np.array(y_test).flatten()
                        predict = predict.flatten()

                        # create the frame
                        act_pred_frame = pd.DataFrame(
                            data={
                                'Actual': y_test,
                                'Predicted': predict
                            }
                        )

                        # create the plot
                        figure = px.scatter(
                            data_frame=act_pred_frame,
                            x='Actual',
                            y='Predicted',
                            title='Actual vs Predicted Values',
                            color='Predicted',
                            color_continuous_scale=px.colors.sequential.GnB
                        )

                        # add the vertical line
                        figure.add_scatter(
                            x=[act_pred_frame['Actual'].min(), act_pred_frame['Actual'].max()],
                            y=[act_pred_frame['Predicted'].min(), act_pred_frame['Predicted'].max()],
                            mode='lines',
                            line=dict(color='#808080',
                                    dash='dash',
                                    width=2),
                            name='Ideal Fit Line'
                        )

                        # update plot background and remove color scale
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
            







# Reactive Calcs for Our Linear Regression Model
@reactive.Calc
def create_testing_training_data():
     """
     Create our Training & Testing data with SkLearn train_test_split function
     """
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(input.lr_test_size()/100), random_state=2792)
     
     return X_train, X_test, y_train, y_test
 
@reactive.calc    
def linear_regression_model():
    
    # Import our training and testing variables
    X_train, X_test, y_train, y_test = create_testing_training_data()
    
    """
    Create the model
    """
    match input.lr_reg_selec():
        case 'none':
            lr = LinearRegression()
        case 'ridge':
            lr = Ridge(alpha=1.0)
        case 'lasso':
            lr = Lasso(alpha=1.0)
        case 'elastic':
            lr = ElasticNet(alpha=1.0, l1_ratio=0.5)
        case _:
            lr = 'none'
    
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


# Reactive Calcs for Our KNN Model
def knn_create_testing_training_data():
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2792)
    
    return X_train, X_test, y_train, y_test

def knn_model():
    
    # get testing and training data
    X_train, X_test, y_train, y_test = knn_create_testing_training_data()
    
    # create model
    knn_model = KNeighborsRegressor(n_neighbors=input.knn_neigh())
    
    # train model
    knn_model.fit(X=X_train, y=y_train)
    
    # make prediction
    prediction = knn_model.predict(X=X_test)
    
    # measure accuracy
    r2 = r2_score(y_true=y_test, y_pred=prediction)
    rmsq = root_mean_squared_error(y_true=y_test, y_pred=prediction)
    median_abs_err = median_absolute_error(y_true=y_test, y_pred=prediction)
    
    print(f"KNN R2 Score: {r2}")
    print(f"KNN Root Mean Sqaured Error: {rmsq}")
    print(f"KNN Median Absolute Error: {median_abs_err}")
    
    neighbors = range(1, 21)
    scores = []

    for k in neighbors:
        knn = KNeighborsRegressor(n_neighbors=k)
        score = cross_val_score(knn, X, y, cv=5, scoring='r2').mean()
        scores.append(score)

    optimal_k = neighbors[scores.index(max(scores))]
    print(f"Optimal k: {optimal_k}")
    
    return prediction, r2, rmsq, median_abs_err

   
# Reactive Calcs for our Decision Tree Model
def reg_tree_training_testing_data():
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=input.reg_tree_test_size(), random_state=2792)
    
    return X_train, X_test, y_train, y_test

def reg_tree_model():
    
    # get training and testing data
    X_train, X_test, y_train, y_test = reg_tree_training_testing_data()
    
    # create model
    reg_tree = DecisionTreeRegressor(criterion=input.reg_tree_criterion(),
                                     splitter='best',
                                     max_depth=input.reg_tree_depth(),
                                     min_samples_leaf=input.reg_tree_min_leaf_samples(),
                                     )
    
    # train the model
    reg_tree.fit(X=X_train, y=y_train)
    
    # make prediction
    prediction = reg_tree.predict(X=X_test)
    
    # get scores
    r2 = r2_score(y_true=y_test, y_pred=prediction)
    mse = mean_squared_error(y_true=y_test, y_pred=prediction)
    mae = mean_absolute_error(y_true=y_test, y_pred=prediction)
    
    return X_train, X_test, y_train, y_test, reg_tree, prediction, r2, mse, mae


# Reactive calcs for our Random Forest Model
def rf_training_testing_data():
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=input.rf_testsize(), random_state=2792)
    
    return X_train, X_test, y_train, y_test

def rf_model():
    
    # get our testing and training data
    X_train, X_test, y_train, y_test = rf_training_testing_data()
    
    # create our model
    rf = RandomForestRegressor(
        n_estimators=input.rf_num_trees(),
        max_depth=input.rf_tree_depth(),
        criterion='squared_error',
        oob_score=True,
        random_state=2792
        )
    
    # train the model
    rf.fit(X=X_train, y=y_train)
    
    # make predictions
    predict = rf.predict(X=X_test)
    
    # get score
    oob_score = rf.oob_score_
    r2 = r2_score(y_true=y_test, y_pred=predict)
    mae = mean_absolute_error(y_test, predict)
    mse = mean_squared_error(y_test, predict)
    
    return rf, predict, oob_score, r2, mae, mse





