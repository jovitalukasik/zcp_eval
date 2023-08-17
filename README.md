# An Evaluation of Zero-Cost Proxies - from Neural Architecture Performance to Model Robustness [[PDF](https://arxiv.org/pdf/2307.09365.pdf)]

Jovita Lukasik, Michael Moeller, Margret Keuper


### Prerequisites


Python > 3.9 and the packages in _requirements.txt_ are needed.

Install packages with:

```
$ pip install -r requirements.txt
```

*Also needed:* 
* download [robustness-data](https://uni-siegen.sciebo.de/s/aFzpxCvTDWknpMA) into ```robustness-data/```
* download [zcp_data](https://drive.google.com/file/d/1R7n7GpFHAjUZpPISzbhxH0QjubnvZM5H/view?usp=share_link) into ````zcp_data````
* download [robustness_dataset_zcp](https://drive.google.com/file/d/1byp12_hWJncd9e-JMYc9Zn0PYGwMOqHg/view?usp=drive_link) into ````robustness_dataset_zcp````


### Run code

To run random forest for the all objectives: **clean, pgd@Linf_eps=1.0, clean-pgd@Linf_eps=1.0** for image data **CIFAR-10** run:

```
$ python random_forest.py --image_data cifar10  --regression_target_2 "fgsm@Linf, eps=1.0"
```

This code also generates the plots from the paper. 


## Citation
```bibtex


@article{lukasik2023,
  author    = {Jovita Lukasik and
               Michael Moeller and
               Margret Keuper},
  title     = {An Evaluation of Zero-Cost Proxies -- from Neural Architecture Performance to Model Robustness},
  journal   = {accepted at GCPR},
  year      = {2023},
}

```

