# CIRH
Work Together: Correlation-Identity Reconstruction Hashing for Unsupervised Cross-modal Retrieval


### Recommendation
We recommend the following operating environment:
- Python == 3.6.x
- Pytorch == 1.8.1 + cu102
- Torchvision == 0.9.1 +cu102
- And other packages

### Datasets
We release the three experimental datasets as follows:
- MIRFlickr, [[Baidu Pan](https://pan.baidu.com/s/1Hm-BFv0epUpJhHJMPkyKsA), password: g8wv]
- NUS-WIDE, [[Baidu Pan](https://pan.baidu.com/s/1QnjYIp-TD5ucmWgrvJFQwg), password: 0lzq]
- MS COCO, [[Baidu Pan](https://pan.baidu.com/s/1fc2h_ow9RajV1oUdo6c1sw), password: s49e]

Please download this three datasets, and put them in the `./datasets/` folder.


### Demo 
Taking MIR Flickr as an example, our model can be trained and verified by the following command:
```bash
python main_mir.py
```

