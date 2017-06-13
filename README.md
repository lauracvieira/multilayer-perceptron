# multilayer-perceptron
Multilayer Perceptron to recognize characters using:
- Cross Validation to train the network
- Multilayer Perceptron as structure
- Python3 as language
- HOG and LBP as image descriptors
- Pyplot as image plotter
- SQLite to store the image descriptors

To run the program:

1- Set parameters on src/parameters.py
2- python3 databases.py (to generate the image descriptors and store in SQLite)
python3 src/cross-validation.py HOG/LBP (to run the program with 3 letters - S, X, Z using HOG or LBP)
3- optional parameter: part2
python3 src/cross-validation.py HOG/LBP (to run the program with 26 letters - A to Z)


by Laura Castro Vieira
Rafael Bortman
Virgílio Fernandes Junior
 
