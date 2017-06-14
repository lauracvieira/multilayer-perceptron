# multilayer-perceptron
Multilayer Perceptron to recognize characters using:</br>
- Cross Validation to train the neural network
- Python 3 as programming language
- HOG and LBP as image descriptors (from Scikit-Image package)
- PyPlot as image plotter (from MatPlotLib package)
- SQLite to store the image descriptors

Before running the program:</br>
- Install <strong>Python 3</strong> and the following packages:</br>
    # MatPlotLib
- pip install -U matplotlib
    # Numpy
- pip install -U numpy
    # Pandas
- pip install -U pandas
    # ProgressBar
- pip install -U progressbar2
    # Scikit-Image
- pip install -U scikit-image
    # Scikit-Learn
- pip install -U scikit-learn
    # Seaborn 
- pip install -U seaborn

To run the program:</br>
- Set parameters on src/parameters.py
- python3 databases.py (to generate the descriptors and store them)
- python3 src/cross-validation.py HOG/LBP (to run the program with 3 letters - S, X, Z using HOG or LBP)
- python3 src/cross-validation.py HOG/LBP part2 (to run the program with 26 letters - A to Z - optional parameter)

Authors:</br>
Laura Castro Vieira</br>
Rafael Bortman</br>
Virgílio Fernandes Junior</br>
 
