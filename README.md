# MNIST Fashion

To start training run:
```python
python mlops/train_model.py
```

To make predictions:
```python
python mlops/predict_model.py [CHECKPOINT_PATH] [DATA_PATH]
```
where `DATA_PATH` has to be a `.pt` file containing loaded images.

To visualize the weights in the 4th fully connected layer in a 2d space:
To make predictions:
```python
python mlops/visualizations/visualize.py
```