# multilayer-perceptron
Multilayer Perceptron to recognize characters using:
- Cross Validation to train the network
- Python3 as language
- HOG and LBP as image descriptors
- Pyplot as image plotter
- SQLite to store the image descriptors

To run the program:

- Set parameters on src/parameters.py
- python3 databases.py (to generate the descriptors and store them)
- python3 src/cross-validation.py HOG/LBP (to run the program with 3 letters - S, X, Z using HOG or LBP)
- python3 src/cross-validation.py HOG/LBP part2 (to run the program with 26 letters - A to Z - optional parameter)


by Laura Castro Vieira
Rafael Bortman
Virgílio Fernandes Junior
 
