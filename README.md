# latefusion

## Descriptions

[Plant identification using score-based fusion of multi-organ images](https://ieeexplore.ieee.org/document/8119457/)

Plant Identification using combinations of multi-organ images. Fusion schemes are max scores, sum scores, product scores, classification based SVM and my Robust Hybrid Model. I  draw a cumulative match characteristic (CMC) curve in order to compare them. Besides that, this project also includes a pretrained AlexNet model.

	
	|__alexnet: AlexNet model to predict vector score for each single organ.
	|
	|__plant_data: contains plant dataset: leaf, flower, branch, entire. We use 50 species from http://www.imageclef.org/lifeclef/2015/plant dataset. It is too big so I can not push it all here. If you are interested in it, do not hesitate to contact me at binhtd.hust@gmail.com.
	|
	|__fusion_data
	|  |__single_organ_score: contains vector score for each single organ.
	|  |	
	|  |__leaf_flower_50_species: contains vector score for each single organ. But each pair of 2 organs that choosen to combine has same id. Each file has format of content: <image id> <species id> <species id from 1-50> <species score equivalently>
	|
	|__fusion_two_organs.ipynb: combine leaf-flower, flower-entire, entire-leaf, branch-leaf, branch-flower, branch-entire in order to increase the accuracy of plant indentification task.
	
## Getting Started

### Data
I used 50 species leaf, flower, branch, entire dataset from http://www.imageclef.org/lifeclef/2015/plant . It is too big, so I can not push it all here. If you are interested in it, do not hesitate to contact me at binhtd.hust@gmail.com.
### Prerequisites
* python 2.7
* tensorflow 0.12.1: https://www.tensorflow.org/versions/r0.12/get_started/os_setup#download-and-setup
* sklearn 0.18.1: http://scikit-learn.org/stable/
### Installing
Firstly, we use AlexNet to export vector score for each single organ:

	(1) ./alexnet/python alexnet_50_species.py --organ leaf
	
	(2) ./alexnet/python alexnet_50_species.py --organ flower
	
	(3) ./alexnet/python alexnet_50_species.py --organ entire
	
	(4) ./alexnet/python alexnet_50_species.py --organ branch

Then, we combine each pair of organ (leaf-flower, flower-entire, ...):

	(5) Open ipython notebook
	
	(6) Open fusion_two_organs.ipynb
	
	(7) Run each block in the notebook. 
	
	Note that: at block #2, replace (organ_1st = 'branch' and organ_2nd = 'entire') by which pair of organ you want to fusion.
	
## Built With

* bvlc_alexnet.npy is the AlexNet's parameters that pre-trained in the ImageNet dataset. You can download it at http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

## Results
My late fusion method (RHF) shows the best performance with highest accuracy rate.

## References

## Authors

**Binh Do** 

## License

This project is licensed under the MIT License
