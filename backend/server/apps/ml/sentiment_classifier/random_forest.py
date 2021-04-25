import joblib
import pandas as pd

class RandomForestClassifier:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.values_fill_missing = joblib.load(path_to_artifacts + "train_mode.joblib")
        self.model = joblib.load(path_to_artifacts + "random_forest.joblib")
        df = pd.read_excel("../../data/data2020.xlsx");
        self.columns = [c for c in df.columns if c != 'Текст' and c != 'Тональность']

    def preprocessing(self, input_data):
        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])
        # fill missing values
        input_data.fillna(self.values_fill_missing)

        for col in input_data:
            if col not in self.columns:
                input_data = input_data.drop(col, 1)

        if (len(input_data) == 1):
            raise ValueError('Попробуйте ввести другой текст')

        for col in self.columns:
            if col not in input_data:
                input_data[col] = 0

        return input_data

    def predict(self, input_data):
        return self.model.predict_proba(input_data)

    def postprocessing(self, input_data):
        label = "low"
        if input_data[0] == 1:
            label = "mid"
        elif input_data[0] == 2:
            label = 'high'
        return {"probability": input_data[0], "label": label, "status": "OK"}

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]  # only one sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction