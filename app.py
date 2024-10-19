from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load the dataset and train the model (this is simplified for the example)
student = pd.read_csv('student_performance.csv')

# Prepare the features and target variable
X = student.drop(['Student_name', 'Overall_Score'], axis=1)
y = student['Overall_Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict on the entire dataset for ranking purposes
predicted_scores = model.predict(X)

# Add the predicted scores to the original dataframe for ranking
student['Predicted_Score'] = predicted_scores

# Sort the students based on the predicted scores (descending order)
student_ranked = student.sort_values(by='Predicted_Score', ascending=False)

# Group by 'Year_Of_Admission' and get the top 3 students for each year
top_students = student_ranked.groupby('Year').head(3)

@app.route('/')
def index():
    return render_template('index.html', top_students=top_students)

@app.route('/year/<int:year>')
def year_results(year):
    # Filter the top students for the selected year
    year_students = top_students[top_students['Year'] == year].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Ensure that the Rank column is calculated
    year_students['Rank'] = year_students['Predicted_Score'].rank(method='first', ascending=False)

    return render_template('results.html', top_students=year_students)

@app.route('/results')
def view_all_results():
    # Get top students per year
    top_students_per_year = student.groupby('Year').apply(lambda x: x.nlargest(3, 'Predicted_Score')).reset_index(drop=True)

    # Create a dictionary to store results by year
    results_by_year = {}
    for year, group in top_students_per_year.groupby('Year'):
        results_by_year[year] = group[['Student_ID', 'Student_name', 'Predicted_Score']].to_dict(orient='records')
    
    # Print the dictionary to check its content and structure
    print(results_by_year)

    return render_template('view-results.html', results_by_year=results_by_year)




if __name__ == '__main__':
    app.run(debug=True)
