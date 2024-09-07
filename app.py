from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the CSV file to get the crop label encoding
file_path = (r"D:\csv_files\Crop_recommendation.csv")
df = pd.read_csv(file_path)
df['label'] = df['label'].astype('category')
label_mapping = dict(enumerate(df['label'].cat.categories))

# Load the trained model
model_filename = (r"D:\sandra\model\crop_model.pkl")
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

def predict_conditions(crop_name):
    if crop_name not in label_mapping.values():
        return f"Crop '{crop_name}' not found in the dataset."
    
    crop_code = [key for key, value in label_mapping.items() if value == crop_name][0]
    input_df = pd.DataFrame([[crop_code]], columns=['label'])
    prediction = model.predict(input_df)
    
    result = {
        'N': prediction[0][0],
        'P': prediction[0][1],
        'K': prediction[0][2],
        'temperature': prediction[0][3],
        'humidity': prediction[0][4],
        'ph': prediction[0][5],
        'rainfall': prediction[0][6]
    }
    
    return result

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input', methods=['GET', 'POST'])
def input():
    if request.method == 'POST':
        crop_name = request.form['crop_name']
        l_crop_name = crop_name.lower()
        conditions = predict_conditions(l_crop_name)
        return render_template('result.html', crop_name=crop_name, conditions=conditions)
    return render_template('input.html')

if __name__ == "__main__":
    app.run(debug=True)
