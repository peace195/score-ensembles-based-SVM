# latefusion

## Descriptions
Plant Identification using combinations of multi-organ images.


	./alexnet: AlexNet model to predict vector score for each single organ.
	
	./plant_data: contains plant dataset: leaf, flower, branch, entire. We use 50 species from http://www.imageclef.org/lifeclef/2015/plant dataset. It is too big so I can not push it all here. If you are interested in it, do not hesitate to contact me at binhdt.hust@gmail.com.
	
	./fusion_data/single_organ_score: contains vector score for each single organ.
	
	./fusion_data/leaf_flower_50_species: contains vector score for each single organ. But each pair of 2 organs that choosen to combine has same id. Each file has format of content: <image id> <species id> <species id from 1-50> <species score equivalently>
	
	./fusion_two_organs.ipynb: combine leaf-flower, flower-entire, entire-leaf, branch-leaf, branch-flower, branch-entire in order to increase the accuracy of plant indentification task.
	
## Getting Started

### Data
Using 50 species leaf, flower, branch, entire dataset from http://www.imageclef.org/lifeclef/2015/plant . It is too big so I can not push it all here. If you are interested in it, do not hesitate to contact me at binhdt.hust@gmail.com.
### Prerequisites
* python 2.7
* tensorflow 0.12.1: https://www.tensorflow.org/versions/r0.12/get_started/os_setup#download-and-setup
* sklearn 0.18.1: http://scikit-learn.org/stable/
### Installing
Firstly, we use AlexNet to export vector score for each single organ:

	(1) ./alexnet/python alexnet_50_species.py leaf
	
	(2) ./alexnet/python alexnet_50_species.py flower
	
	(3) ./alexnet/python alexnet_50_species.py entire
	
	(4) ./alexnet/python alexnet_50_species.py branch

Then, we combine each pair of organ (leaf-flower, flower-entire, ...):

	(5) Open ipython notebook
	
	(6) Open fusion_two_organs.ipynb
	
	(7) Run each block in the notebook. 
	
	Note that: at block #2, replace (organ_1st = 'branch' and organ_2nd = 'entire') by which pair of organ you want to fusion.
	
## Built With

* bvlc_alexnet.npy is the AlexNet's paremeters that pre-trained in ImageNet dataset. You can download it at http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/


## Authors

* **Binh Thanh Do** 

## License

This project is licensed under the MIT License