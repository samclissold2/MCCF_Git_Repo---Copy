import os
import pandas as pd
import json
import ast

def get_files(base_dir, extensions=None):
    file_list = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if not extensions or file.endswith(tuple(extensions)):
                file_list.append(os.path.join(root, file))
    return file_list

def summarize_csv(path, max_rows=3):
    df = pd.read_csv(path, nrows=max_rows)
    return {
        "path": path,
        "columns": df.columns.tolist(),
        "sample_data": df.to_dict(orient='records')
    }

def summarize_python(path):
    with open(path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    return {
        "functions": [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)],
        "classes": [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)],
    }

def save_summary(data, out_file="repo_summary.json"):
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
