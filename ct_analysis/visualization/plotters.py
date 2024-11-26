# Plotting functions

def create_summary_visualizations(evolution_df, output_dir):
    """Create summary visualizations with fixed colorbar"""
    features = [
        'peak_amplitude', 'pattern_width', 'area_under_curve',
        'mean_slope', 'max_slope', 'risk_score'
    ]
    
    # Create one main colorbar for R² values
    colorbar_settings = dict(
        title=dict(
            text='R²',
            side='right'
        ),
        tickformat='.2f',
        len=0.5,  # Length of colorbar
        yanchor='middle',
        y=0.5,    # Center vertically
        xanchor='left',
        x=1.05    # Position to the right of plots
    )
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            f"{feature.replace('_', ' ').title()} Growth Rate vs Initial Value"
            for feature in features
        ],
        vertical_spacing=0.2,
        horizontal_spacing=0.15
    )
    
    row = 1
    col = 1
    for feature in features:
        # Create scatter plot
        scatter = go.Scatter(
            x=evolution_df[f'{feature}_initial_value'],
            y=evolution_df[f'{feature}_growth_rate'],
            mode='markers',
            marker=dict(
                size=8,
                color=evolution_df[f'{feature}_r_squared'],
                colorscale='Viridis',
                showscale=(col == 3 and row == 1),  # Show colorbar only once
                colorbar=colorbar_settings if (col == 3 and row == 1) else None
            ),
            text=[
                f"Location: {row['start_location']:.1f}m - {row['end_location']:.1f}m<br>" +
                f"Global ID: {row['global_id']}<br>" +
                f"R²: {row[f'{feature}_r_squared']:.3f}"
                for _, row in evolution_df.iterrows()
            ],
            hovertemplate="Initial Value: %{x:.2f}<br>" +
                         "Growth Rate: %{y:.2e}/day<br>" +
                         "%{text}<br>" +
                         "<extra></extra>"
        )
        
        fig.add_trace(scatter, row=row, col=col)
        
        # Update axes labels
        fig.update_xaxes(
            title_text=f"Initial {feature.replace('_', ' ').title()}",
            row=row, col=col
        )
        fig.update_yaxes(
            title_text="Growth Rate (per day)",
            row=row, col=col
        )
        
        col += 1
        if col > 3:
            col = 1
            row += 1
    
    fig.update_layout(
        height=900,
        width=1500,
        title='Growth Rate Analysis of Track Features',
        showlegend=False,
        template='plotly_white',
        margin=dict(r=150)  # Make room for colorbar
    )
    
    fig.write_html(os.path.join(output_dir, 'growth_rate_analysis.html'))


    def plot_spectral_heatmap(self, data, feature, output_dir):
        """Create a heatmap showing feature evolution with Island IDs"""
        # Prepare data
        pivot_data = data.pivot(
            index='date',
            columns='global_id',
            values=feature
        )
        
        # Get location information for each island
        island_info = data.groupby('global_id').agg({
            'start_location': 'mean',
            'end_location': 'mean'
        }).sort_values('start_location')
        
        # Sort islands by location
        pivot_data = pivot_data.reindex(columns=island_info.index)
        
        # Create x-axis labels with Island IDs and locations
        x_labels = [
            f"IS:{id}<br>{start:.0f}-{end:.0f}m" 
            for id, (start, end) in zip(
                island_info.index, 
                island_info[['start_location', 'end_location']].values
            )
        ]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=x_labels,
            y=pivot_data.index,
            colorscale='Viridis',
            colorbar=dict(title=feature)
        ))
        
        # Update layout for visual adjustments
        fig.update_layout(
            title=f'Temporal Evolution of {feature} Across Track',
            xaxis_title='Island ID and Location',
            yaxis_title='Date',
            height=800,
            xaxis=dict(
                tickangle=90,
                tickmode='array',
                ticktext=x_labels,
                tickvals=list(range(len(x_labels))),
                showgrid=False  # Disable x-axis gridlines
            ),
            yaxis=dict(
                showgrid=False  # Disable y-axis gridlines
            ),
            paper_bgcolor='black',  # Set the overall background color
            plot_bgcolor='black',  # Set the plot's background color
            font=dict(color='white')  # Adjust font color for visibility on black
        )
        
        fig.write_html(os.path.join(output_dir, f'spectral_evolution_{feature}.html'))

    def plot_island_evolution(self, island_data, global_id, output_dir):
        """Create evolution plot with comprehensive trend analysis and predictions"""
        try:
            island_data = island_data.copy()  # Make a copy to prevent modifications
            island_data = island_data.sort_values('date')
            dates = pd.to_datetime(island_data['date'])
            
            # Create subplots for each feature
            fig = make_subplots(
                rows=len(self.features_to_track), 
                cols=1,
                subplot_titles=[f'{feature.replace("_", " ").title()} Evolution' 
                            for feature in self.features_to_track],
                vertical_spacing=0.08
            )
            
            start_loc = float(island_data['start_location'].mean())  # Explicitly convert to float
            end_loc = float(island_data['end_location'].mean())  # Explicitly convert to float
            
            for i, feature in enumerate(self.features_to_track, 1):
                try:
                    # Convert values to float explicitly
                    values = island_data[feature].astype(float).values
                    
                    # Plot actual values
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=values,
                            mode='markers+lines',
                            name=f'Actual {feature.replace("_", " ").title()}',
                            marker=dict(size=8),
                            line=dict(width=2)
                        ),
                        row=i, col=1
                    )
                    
                    # Linear regression for trend analysis
                    days = (dates - dates.min()).dt.total_seconds() / (24 * 3600)
                    days = days.astype(float)
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(days, values)
                    r_squared = r_value ** 2
                    
                    # Generate trend line
                    future_days = np.linspace(0, days.max() * 2, 100)
                    trend_dates = dates.min() + pd.Timedelta(days=float(future_days.max()))
                    trend_values = slope * future_days + intercept
                    
                    # Add LSTM predictions specifically for risk_score
                    if (feature == 'risk_score' and 
                        self.risk_predictor is not None and 
                        len(island_data) >= self.risk_predictor.sequence_length):
                        try:
                            predictions = self.risk_predictor.predict_trajectory(island_data)
                            if len(predictions) > 0:
                                # Generate future dates for predictions
                                last_date = dates.max()
                                future_dates = pd.date_range(
                                    start=last_date + pd.Timedelta(days=1),
                                    periods=len(predictions),
                                    freq='D'
                                )
                                
                                # Add prediction line
                                fig.add_trace(
                                    go.Scatter(
                                        x=future_dates,
                                        y=predictions,
                                        mode='lines',
                                        name='LSTM Predictions',
                                        line=dict(
                                            dash='dash',
                                            color='red',
                                            width=2
                                        )
                                    ),
                                    row=i, col=1
                                )
                                
                                # Add uncertainty ribbon if available
                                if hasattr(self.risk_predictor, 'get_prediction_uncertainty'):
                                    lower_bound, upper_bound = self.risk_predictor.get_prediction_uncertainty(predictions)
                                    fig.add_trace(
                                        go.Scatter(
                                            x=future_dates,
                                            y=upper_bound,
                                            mode='lines',
                                            line=dict(width=0),
                                            showlegend=False
                                        ),
                                        row=i, col=1
                                    )
                                    fig.add_trace(
                                        go.Scatter(
                                            x=future_dates,
                                            y=lower_bound,
                                            mode='lines',
                                            line=dict(width=0),
                                            fillcolor='rgba(255, 0, 0, 0.2)',
                                            fill='tonexty',
                                            name='95% Confidence'
                                        ),
                                        row=i, col=1
                                    )
                        except Exception as e:
                            print(f"Error adding predictions for island {global_id}: {str(e)}")
                    
                    # Update axes labels
                    fig.update_xaxes(
                        title_text="Date",
                        tickformat="%Y-%m-%d",
                        tickangle=45,
                        row=i, col=1
                    )
                    fig.update_yaxes(
                        title_text=feature.replace("_", " ").title(),
                        row=i, col=1
                    )
                    
                except Exception as e:
                    print(f"Error processing {feature} for island {global_id}: {str(e)}")
                    continue
            
            # Update layout
            fig.update_layout(
                height=250 * len(self.features_to_track),
                title=f'Evolution of Island {global_id}<br>(Location: {start_loc:.1f}m - {end_loc:.1f}m)',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05
                ),
                margin=dict(r=250, b=50)
            )
            
            # Save plot
            try:
                fig.write_html(os.path.join(output_dir, f'evolution_island_{global_id}.html'))
            except Exception as e:
                print(f"Error saving plot for island {global_id}: {str(e)}")
                
        except Exception as e:
            print(f"Error in plot_island_evolution for island {global_id}: {str(e)}")
