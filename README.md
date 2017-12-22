# **Traffic light classifer**

### Objective
Use a nerual network to predict the state of a traffic lights, e.g. "red", "yellow", "green", "off"


#### Current accuracy:

94.2% at Bosch dataset, Udacity simular and udacity car dataset.




#### The state definition of the traffic lights are as follow:

| Traffic light state 	| red 	| yellow 	| green 	| off 	|
|:-------------------:	|:---:	|:------:	|:-----:	|:---:	|
| Index 	| 1 	| 2 	| 3 	| 4 	|



#### How to run the demo

```sh

python main.py

```

#### How to run with one example

![alt text][green]

Use the code in the `main.py`
```sh
    file_path = './data/green.jpg'
    predicted_state = test_an_image(file_path, model=load_model('model.h5'))
```

Predicted state of the traffic light:

```sh
green
```

[green]: ./data/green.jpg
