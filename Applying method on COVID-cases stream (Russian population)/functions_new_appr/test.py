import pandas as pd
import pickle

def test_relative_paths():
    with open(r"temporary_files_new_appr/file.pkl","wb") as file:
        pickle.dump(pd.DataFrame([1,2]),file)