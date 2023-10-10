# Joint Out-of-Distribution Detection and Uncertainty Estimation for Trajectory Prediction

This is the official project page including the paper and the code of JOODU:

**[Joint Out-of-Distribution Detection and Uncertainty Estimation for Trajectory Prediction](https://arxiv.org/abs/2308.01707)**
<br>
Julian Wiederer, Julian Schmidt, Ulrich Kressel, Klaus Dietmayer, Vasileios Belagiannis
<br>
*Accepted for presentation at [IROS 2023](https://ieee-iros.org/) in Detroit, MI, US.*
<br>

<!---![joodu](./images/method.png)--->

<div align="center">
<img src="https://github.com/againerju/joodu/blob/master/images/method.png" width = 80% height = 80%>
</div>

## Citation

```bibtex
@InProceedings{wiederer2023joodu,
  author={Julian Wiederer and Julian Schmidt and Ulrich Kressel and Klaus Dietmayer and Vasileios Belagiannis},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Sytems (IROS)}, 
  title={Joint Out-of-Distribution Detection and Uncertainty Estimation for Trajectory Prediction}, 
  year={2023}}
```

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">
<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png"
 /></a><br />JOODU is licensed under <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"
 >Creative Commons Attribution-NonCommercial 4.0 International License</a>.
 
Check [LICENSE](LICENSE) for more information.

## Installation
The following section describes the setup of the environment and the installation of the required packges including the Shifts and the Argoverse API.

### Conda Environment
We recommend using Anaconda.
The installation is described on the following page:\
https://docs.anaconda.com/anaconda/install/linux/

Next, install the required packages
```sh
conda env create -f environment.yml
```

and activate the environment
```sh
conda activate joodu
```

### Install Shifts API
Install the API for the Shifts Vehicle Motion Prediction Dataset as described here:\
https://github.com/Shifts-Project/shifts/tree/main/sdc

### Install Argoverse API
The argoverse-api is used to convert the HD-map provided by Shifts into the Argoverse format, which is consumed by the trajectory prediction model.

```
pip install git+https://github.com/argoai/argoverse-api.git
```

## Setup Shifts Dataset

Donwload and extract the datasets

```
chmod +x scripts/fetch_shifts_dataset.sh
./scripts/fetch_shifts_dataset.sh
```

The data is pre-processed online during the evaluation.

## Model
Download the model parameters 

```
chmod +x scripts/download_model.sh
./download_model.sh
```

## Train Model
Under construction - coming soon.

## Test Model on Evaluation Set


The following command runs the evaluation on the Shifts eval set.\
In case you run the evaluation for the first time, the data pre-processing is performed online.

```
chmod +x scripts/eval.sh
./scripts/eval.sh
```


