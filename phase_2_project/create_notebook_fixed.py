"""Create a proper Jupyter notebook from the Python script."""
import json

# Read the Python script
with open('train_ml_colab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Split by cell markers
blocks = content.split('# %%')

# Create notebook structure
notebook = {
    'cells': [],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.8.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 2
}

# Process each block
for block in blocks:
    if not block.strip():
        continue
    
    lines = block.split('\n')
    
    # Check if it's markdown
    is_markdown = '[markdown]' in lines[0]
    
    if is_markdown:
        # Remove the [markdown] marker
        lines[0] = lines[0].replace('[markdown]', '').strip()
        
        # Process markdown: remove leading '# ' from each line
        source_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                if line.startswith('# '):
                    source_lines.append(line[2:])
                else:
                    source_lines.append(line)
            else:
                source_lines.append('')
        
        # Remove empty lines at start and end
        while source_lines and not source_lines[0].strip():
            source_lines.pop(0)
        while source_lines and not source_lines[-1].strip():
            source_lines.pop()
        
        if not source_lines:
            continue
        
        cell = {
            'cell_type': 'markdown',
            'metadata': {},
            'source': source_lines
        }
    else:
        # Code cell - keep all lines as-is
        source_lines = []
        for line in lines:
            if line == lines[0] and not line.strip():  # Skip first empty line
                continue
            source_lines.append(line)
        
        # Remove trailing empty lines
        while source_lines and not source_lines[-1].strip():
            source_lines.pop()
        
        if not source_lines:
            continue
        
        cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': source_lines
        }
    
    notebook['cells'].append(cell)

# Save notebook
with open('train_ml_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"✓ Created notebook with {len(notebook['cells'])} cells")
print(f"  - Markdown cells: {sum(1 for c in notebook['cells'] if c['cell_type'] == 'markdown')}")
print(f"  - Code cells: {sum(1 for c in notebook['cells'] if c['cell_type'] == 'code')}")

