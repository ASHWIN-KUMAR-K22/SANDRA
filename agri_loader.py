import pandas as pd
import pickle

# Load the CSV file to get the crop label encoding
file_path = (r"D:\csv_files\Crop_recommendation.csv")
df = pd.read_csv(file_path)
df['label'] = df['label'].astype('category')
label_mapping = dict(enumerate(df['label'].cat.categories))

# Load the trained model
model_filename = 'crop_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

def predict_conditions(crop_name):
    # Check if the crop name exists in the label mapping
    if crop_name not in label_mapping.values():
        return f"I don't know about this Crop '{crop_name}' ."
    
    # Get the encoded crop label
    crop_code = [key for key, value in label_mapping.items() if value == crop_name][0]
    
    # Create a DataFrame for the input with the same column name
    input_df = pd.DataFrame([[crop_code]], columns=['label'])
    
    # Make a prediction
    prediction = model.predict(input_df)
    
    # Prepare the output
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

if __name__ == "__main__":
    while True:
        crop_name = input("Enter the crop name (or type 'exit' to quit): ").strip()
        if crop_name.lower() == 'exit':
            print("Exiting the program.")
            break
        conditions = predict_conditions(crop_name)
        if isinstance(conditions, str):
            print(conditions)
        else:
            print(f"Predicted suitable conditions for {crop_name}:")
            for key, value in conditions.items():
                print(f"{key}: {value:.2f}")
