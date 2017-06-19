# latefusion

## Descriptions
Plant Identification using combinations of multi-organ images.

	./alexnet: AlexNet model to predict vector score for each single organ.
	
	./plant_data: contains plant dataset: leaf, flower, branch, entire. We use 50 species from http://www.imageclef.org/lifeclef/2015/plant dataset.
	
	./fusion_data/single_organ_score: contains vector score for each single organ.
	
	./fusion_data/leaf_flower_50_species: contains vector score for each single organ. But each pair of 2 organs that choosen to combine has same id. Each file has format of content: <image id> <species id> <species id from 1-50> <species score equivalently>
	
	...
## Getting Started

### Data

### Prerequisites
* python 2.7
* tensorflow 0.12.1: https://www.tensorflow.org/versions/r0.12/get_started/os_setup#download-and-setup
* sklearn 0.18.1: http://scikit-learn.org/stable/
### Installing

## Built With

* bvlc_alexnet.npy is the AlexNet's paremeters that pre-trained in ImageNet dataset. You can download it at http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/


## Authors

* **Binh Thanh Do** 

## License

This project is licensed under the MIT License