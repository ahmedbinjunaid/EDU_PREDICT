from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from flask import send_file
from werkzeug.utils import secure_filename
import io


app = Flask(__name__)

# Load trained models
final_score_model = joblib.load('models/final_score_model.pkl')
dropout_model = joblib.load('models/dropout_model.pkl')
demand_model = joblib.load('models/demand_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Match feature order with training
        feature_names = [
            'Age', 'Gender', 'Location', 'Socioeconomic_Status', 'Parent_Education_Level',
            'Previous_Semester_GPA', 'Midterm_Score', 'Assignment_Average', 'Backlogs_Count',
            'Attendance_Percentage', 'Class_Skips', 'Late_Submissions_Count',
            'Avg_Weekly_Logins', 'Video_Watch_Time', 'Assignments_Submitted',
            'Discussion_Participation', 'Semester', 'Course_ID', 'Device_Used_To_Login', 'Feedback_Score'
        ]
        form_values = [float(request.form.get(name)) for name in feature_names]
        features = np.array(form_values).reshape(1, -1)

        # Predict
        final_score = final_score_model.predict(features)[0]
        dropout = dropout_model.predict(features)[0]
        demand = demand_model.predict(features)[0]

        return render_template('index.html',
                               final_score=round(final_score, 2),
                               dropout='Yes' if dropout == 1 else 'No',
                               demand=round(demand, 2),
                               submitted=True)
    return render_template('index.html', submitted=False)

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    prediction_df = None
    error = None

    if request.method == 'POST':
        try:
            file = request.files['file']
            if not file.filename.endswith('.csv'):
                raise ValueError("Only .csv files are supported")

            df = pd.read_csv(file)

            expected_columns = [
                'Age', 'Gender', 'Location', 'Socioeconomic_Status', 'Parent_Education_Level',
                'Previous_Semester_GPA', 'Midterm_Score', 'Assignment_Average', 'Backlogs_Count',
                'Attendance_Percentage', 'Class_Skips', 'Late_Submissions_Count',
                'Avg_Weekly_Logins', 'Video_Watch_Time', 'Assignments_Submitted',
                'Discussion_Participation', 'Semester', 'Course_ID', 'Device_Used_To_Login', 'Feedback_Score'
            ]

            if not all(col in df.columns for col in expected_columns):
                raise ValueError("CSV missing required columns")

            X = df[expected_columns]

            df['Predicted_Final_Score'] = final_score_model.predict(X)
            df['Predicted_Dropout'] = dropout_model.predict(X)
            df['Predicted_Demand'] = demand_model.predict(X)

            prediction_df = df.to_html(classes='table table-bordered', index=False)

            # Save to downloadable CSV
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            with open("static/predicted_batch.csv", "w", encoding="utf-8") as f:
                f.write(output.read())

        except Exception as e:
            error = str(e)

    return render_template("batch.html", prediction_table=prediction_df, error=error)



if __name__ == '__main__':
    app.run(debug=True)
