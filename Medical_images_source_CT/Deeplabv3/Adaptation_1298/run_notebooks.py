import nbformat
from nbconvert import NotebookExporter
from nbconvert.preprocessors import ExecutePreprocessor

# List of notebooks to be run in sequence
notebooks = [
    'Adaptation-step4-test_stride_(1, 1, 1)-[0].ipynb',
    'Adaptation-step4-test_stride_(1, 1, 1)-[1].ipynb',
    'Adaptation-step4-test_stride_(1, 1, 1)-[2].ipynb',
    'Adaptation-step4-test_stride_(1, 1, 1)-[3].ipynb'
]

# Setup the executor
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

# Loop through the notebooks and execute them
for notebook in notebooks:
    with open(notebook, "r", encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        ep.preprocess(nb, {'metadata': {'path': './'}})  # Ensure the path is correct for finding local modules
        with open(notebook, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Finished executing {notebook}")

print("All notebooks executed successfully.")
