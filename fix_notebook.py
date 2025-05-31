import json
import re

# Read the notebook
with open('main.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find and fix cells with the problematic code
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        if isinstance(source, list):
            source_text = ''.join(source)
            
            # Fix the drop statement
            if "df.drop(['Date', 'Adj Close'], axis=1)" in source_text:
                new_source = source_text.replace(
                    "df = df.drop(['Date', 'Adj Close'], axis=1)",
                    """# Check available columns and drop only if they exist
columns_to_drop = []
if 'Date' in df.columns:
    columns_to_drop.append('Date')
if 'Adj Close' in df.columns:
    columns_to_drop.append('Adj Close')

if columns_to_drop:
    df = df.drop(columns_to_drop, axis=1)
else:
    # For MultiIndex columns, just use Close price
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']  # Get just the Close prices"""
                )
                cell['source'] = [new_source]
                # Clear output since it contains errors
                cell['outputs'] = []
                cell['execution_count'] = None

# Save the fixed notebook
with open('main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Fixed main.ipynb")
