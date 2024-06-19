import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        try:
            data = pd.read_excel(self.file_path)
            print(f"Loaded data from {self.file_path} successfully.")
            return data
        except Exception as e:
            print(f"Failed to load data from {self.file_path}. Error: {e}")
            raise e
