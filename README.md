# image-classification-gcp
COMS 6998 Practical Deep Learning System Performance Final Project

Name: Sung-Ping (Josh) Chang, Chia-Jou Kuo

## Description of the Project
In this project, we want to explore the usage and end2end pipline of AutoML for image classifiation task comprehensively. We will compare the performance, cost, potential usage of AutoML with standard VM to make positive impact for developers. We used 2 different dataset (sports_100 and food_101) to train image classification model. Finally, we will deploy our models on the edge device and can inference on the real-word images from users.


## Description of the Repository
This repo is training image classification model for standard VM in order to compare to the same task using Automl. Besides, it contains our ios application xcode project (using Swift and Objective C++ code) to deploy the model on iPhone.


## Example commands to execute the code

People can enter the `foods` or `sports` directory and execute `python3 train.py` to train the image classification model.

## Results and observations  

Performance comparison for AutoML Vision vs VM instance (NVIDIA Tesla V100) (Accuracy, Training Time)

|    AutoML Vision     | Accuracy | Training Time (node hours) | VM instances | Accuracy | Training Time (node hours) |
| :------------------: | :------: | :------------------------: | :----------: | :------: | :------------------------: |
|     Sports-Cloud     |  0.9900  |         20                 |  Resnet 101  |  0.9700  |           3.55             |
|     Sports-edge      |  0.9760  |         3.84               |  MobileNet_v3|  0.9600  |           0.64             |
|     Food-Cloud       |  0.7700  |         20                 |  Resnet 50   |  0.8337  |           18.63            |
|     Food-edge        |  0.8190  |         4.08               |  MobileNet_v3|  0.9600  |           7.02             |


Performance comparison for AutoML vs VM instance (NVIDIA Tesla V100) (Price)

|    AutoML Vision     | Accuracy | Price (per node hour)| Total Price  | VM instances | Accuracy | Price (per node hour)| Total Price  |
| :------------------: | :------: | :------------------: | :----------: | :----------: | :------: | :-----------------:  | :----------: |
|     Sports-Cloud     |  0.9900  |        3.465         |    69.3      |  Resnet 101  |  0.9700  |         1.91         |     6.781    |
|     Sports-edge      |  0.9760  |        18.00         |    69.13     |  MobileNet_v3|  0.9600  |         1.91         |     1.222    |
|     Food-Cloud       |  0.7700  |        3.465         |    69.3      |  Resnet 50   |  0.8337  |         1.91         |     35.58    |
|     Food-edge        |  0.8190  |        18.00         |    73.49     |  MobileNet_v3|  0.8001  |         1.91         |     13.41    |


From this project, we learn that AutoML provides a convenient way to help developers train model easily, and comprehensive evaluation using different metrics. However, AutoML is much more expensive than using standard VM to train a model and is not flexible for custmoize settings. Users should focus on budget and their need to decide which is more suitable for the cases.


