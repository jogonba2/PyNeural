# PyNeural
Make your neural networks with PyNeural.

In PyNeural.py script are implemented two examples of neural networks corresponding to: </br>
<img src="https://i.gyazo.com/27e1802a18be451bef187dc1cc208b24.png"/> </br>
<img src="https://i.gyazo.com/2a011b7b0c42c6a0cd73970c3bacc9a8.png"/>

Currently PyNeural only can be used for classify a sample or make a regression <img src="https://latex.codecogs.com/gif.latex?%5Cmathbb%7BR%7D%5E%7Bd%7D%5Crightarrow%20%5Cmathbb%7BR%7D%5E%7Bd%27%7D%20/%20d%3D%5Cleft%20%7C%20input%5C%3B%20layer%20%5Cright%20%7C%20%2C%20d%27%3D%5Cleft%20%7C%20output%5C%3B%20layer%20%5Cright%20%7C">, computation of weights theta has not been yet implemented, for this reason you must to set theta parameter </br>

To configure a neural network, you must to set the next variables: </br></br>
<img src="https://latex.codecogs.com/gif.latex?x%3Dsample%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%7D"/></br></br>
<img src="https://latex.codecogs.com/gif.latex?theta%3D%5B%5B%5Bweight%5D_%7B0%7D%20...%20%5Bweight%5D_%7Blayerunits%7D%5D%5D_%7B0%7D%20...%20%5B%5Bweight%5D_%7B0%7D%20...%20%5Bweight%5D_%7Blayerunits%7D%5D%5D_%7B%7Clayers%7C%7D%5D%20%5D"></br></br>
<img src="https://latex.codecogs.com/gif.latex?nHiddenLayers%3D%20number%5C%3B%20of%5C%3B%20hidden%5C%3B%20layer%5CM%20%5Cin%20%5Cmathbb%7BN%7D"></br></br>
<img src="https://latex.codecogs.com/gif.latex?nUnitsPerLayer%20%3D%20number%5C%3B%20of%5C%3B%20units%5C%3B%20per%5C%3B%20layer%5Cin%20%5Cmathbb%7BN%7D%5E%7Bd%7D"></br></br>
<img src="https://latex.codecogs.com/gif.latex?outputUnits%20%3D%20number%5C%3B%20of%5C%3B%20hidden%5C%3B%20units%5C%3B%20in%5C%3B%20output%5C%3B%20layer%20%5Cin%20%5Cmathbb%7BN%7D"></br></br>
<img src="https://latex.codecogs.com/gif.latex?fActivate%20%3D%20activation%5C%3B%20function%5C%3B%20for%5C%3B%20each%5C%3B%20neuron%5C%3B%20of%5C%3B%20the%5C%3B%20neural%5C%3B%20network%5Cin%20%5Cbegin%7Bbmatrix%7D%20lineal%20%2C%26%20jump%2C%20%26%20sigmoid%2C%20%26%20hiperbolic%5C%3B%20tangent%2C%20%26%20fast%20%5Cend%7Bbmatrix%7D"></br></br>



