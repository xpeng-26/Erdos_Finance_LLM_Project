# Contribution
Thank you for considering contributing to the Erdos Finance LLM Project! We welcome contributions from everyone. Please be aware of the contribution principles to help us maintain the codebase.

## Principles
- Follow the existing code style.
- Keep the existing file structures.
- Write clear, concise commit messages.
- Include comments and documentation where necessary.

## Setup
Here are some guidelines to help you get started:

1. **Python Version**: Make sure to use 3.13.0 under the project path.
	```bash
	# change the directory to your project path in the terminal
	# if you have multiple versions of python installed and the default one is not 3.13.0, you will need to specify it
	poetry env use <path>/python3.13
	# check the python version
	python --version
	# should be Python 3.13.0
	```

2. **Install Poetry**: We use [Poetry](https://python-poetry.org/) for dependency (python package version) management.
	```bash
	pip install poetry

	# check if successfully installed
	poetry -V
	```

3. **Install Python packages via Poetry**: All packages in the `pyproject.toml` file will be installed in the virtual environment path (not in the project path). This feature can benefit from duplicated package files across different projects used with Poetry.
	```bash
	poetry install --no-root

	# check if the packages have been successfully installed
	poetry show
	```

4. **Test the environment**

	You can test the environment in one of three following ways:

	(1) Run the Python file directly:
		```
		poetry run python tests/test_env/test_/test.py
		```

	(2) Run the bash shell file which controls the Python file:
		```
		bash tests/test_env/run_test_env.sh 
		```

	(3) Run the test.py file in the Jupyter notebook `tests/test_env/test.ipynb`

	You can check [here](#configure-the-experiment-and-run-scripts) for more details.

## Install Some Special Dependents
1. `TA-Lib`

	Since, `TA-Lib` requires a system-level library, we cannot just use Poetry to install the ta-lib. Instead, we should
	use Brew to install it in the local first, and then add to the poetry
	project environment.
	```bash
	brew install ta-lib
	# add to poetry project environment
	poetry add ta-lib
	```

2. `llama.cpp`

	Building `llama.cpp` with Metal (Apple M Chip) support and shared library enabled.
	- The following steps are used to build the `llama.cpp` locally to support several projects. 
	- If you just want to touch up it for this project and want to install the pre-build version, try to use official binding tool `llama-cpp-python` or `brew install llama.cpp`.

	Step 1. Download `llama.cpp`

	```bash
	LLAMA_DIR = <Path> # used to store llama.cpp and models
	# LLAMA_DIR = ~/Tools/llm_tools
	mkdir -p "$LLAMA_DIR"
	cd "$LLAMA_DIR"
	git clone  https://github.com/ggml-org/llama.cpp
	cd llama.cpp 
	```
	
	Step 2. Build the wheel

	```bash
	mkdir build && cd build

	# install cmake if not installed
	# brew install cmake
	cmake -DLLAMA_METAL=on -DBUILD_SHARED_LIBS=on ..
	cmake --build . --config Release
	```

	Step 3. Manage `dylib` file

	```bash
	# the path to manage all versions of .dylib files for furture usage
	LLAMA_LIB_OUTPUT=<Path>
	# LLAMA_LIB_OUTPUT=~/Tools/llm_tools/lib
	mkdir -p "$LLAMA_LIB_OUTPUT"

	cp bin/libllama.dylib "$LLAMA_LIB_OUTPUT"
	```

	Step 4. Download LLM model

	```bash
	# the path to store all different LLMs files
	LLAMA_MODEL=<Path>
	# LLAMA_MODEL=~/Tools/llm_tools/models
	mkdir -p "$LLAMA_MODEL"
	cd "$LLAMA_MODEL"

	# download model
	# make sure to install wget to download large files
	# brew install wget 
	# wget <model.gguf link>

	# some model is gated, which is needed to aggree the liscence
	# better to download via browser
	# eg. to download gemma-3-1b-it-qat-q4_0 from https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf/tree/main
	```

## How to Contribute
1. **Clone the repository to local** and change the directory to the project path.
	```bash
	cd <path>/Erdos_Finance_LLM_Project
	```
2. **Create a new branch**: 
	```bash
	git checkout -b your-branch-name
	```
3. **Pull the newest version from the GitHub server**:
	```bash
	git pull
	```
3. **Make your changes**: Implement your feature or bug fix.
	If you need to install a new package, you will need to use Poetry to install it.
	```bash
	poetry add <package>
	poetry install
	```
4. **Commit your changes**: 
	```bash
	git commit -m "Description of your changes"
	```
6. **Push your changes to the GitHub server**: 
	```bash
	git push origin your-branch-name
	```
7. **Create a Pull Request**: When you have finished all steps of a plan and want to merge your code with ours, please create a Pull Request. Go to the original repository on the GitHub webpage, click "New Pull Request" and **add the description** of your effort.

## Report Issues
To report issues, please use the [GitHub Issues page](https://github.com/your-repo/Finance_LLM_Project/issues). Provide a detailed description of the problem, steps to reproduce it, and any relevant logs or screenshots. This will help us address the issue more efficiently.

# File Structure

The project is organized as follows:

	```
	/ErdosNewsFinanceProject
	├── config/            					 # Experiment setup by TOML files
	├── src/               					 # Source code files
	│   ├── ingestion/     			     # Data ingestion and cleaning
	│   ├── feature/     				     # Model scripts for feature engineering via LLMs
	│   ├── stratergy/    			     # Model scripts for optimizing trading stratergies
	│   ├── evaluation/    			     # Model evaluation
	│   ├── utils/         					 # Utility functions (reusable functions in all other folders)
	│   └── trade_w_llm.py 		       # The main file to call all drivers (steps) in engine folders for trading
	├── scripts/           					 # Shell scripts to run engine driver
	├── notebook/          					 # Jupyter notebook for visualization or small experiments
	├── data/              					 # Data files (small size, core data should store in Google Drive)
	│   ├── figures/       					 # Figure files
	│   └── models/        					 # Model files
	├── tests/             					 # Test files
	├── doc/               					 # Project documentation
	├── .gitignore         					 # Git ignore file
	├── README.md          					 # Project introduction
	├── pyproject.toml     					 # Poetry file to control dependencies
	├── poetry.lock        					 # Poetry file
	└── CONTRIBUTING.md    					 # Contribution guidelines
	```

	This structure helps in maintaining a clean and organized codebase, making it easier to navigate and contribute to the project.

# Configure the Experiment and Run Scripts

**Run Python scripts**

	```bash
	# method 1. run <file>.py script with poetry
	poetry run python <file>.py

	# method 2. open poetry shell
	poetry shell
	## then open python at the shell
	python
	## or run python scripts
	python <file>.py
	```

	If you want to run Python code in the Jupyter Notebook, you will need to choose the kernel as the poetry environment for the current project first. For example, in VS Code, after opening the Jupyter Notebook, you will find the button "Select Kernel" on the top right to choose the kernel.

# Style Guide
