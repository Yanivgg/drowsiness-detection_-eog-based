"""Fix notebook - ensure source is list of lines."""
import json

# Read existing notebook
with open('train_ml_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix each cell
for cell in nb['cells']:
    if isinstance(cell['source'], str):
        # Convert string to list of lines
        cell['source'] = cell['source'].split('\n')
    elif isinstance(cell['source'], list) and len(cell['source']) > 0:
        # If it's a list but joined without newlines, split it
        if len(cell['source']) == 1 and '\n' in cell['source'][0]:
            cell['source'] = cell['source'][0].split('\n')

# Save fixed notebook
with open('train_ml_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("✓ Notebook fixed - all cells now have proper line breaks")
print(f"Total cells: {len(nb['cells'])}")

