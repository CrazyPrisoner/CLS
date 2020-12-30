# CLS

# Install reqirements

    pip install -r reqirements.txt
    
# Train CNN model

    python cnn_model.py --training_iteration=10 --model_version=1 ./models
    output: /models folder in your local directory

# Running test in scpit

## First need to run Tensorflow Serving

### Running TS as service on linux
   
    tensorflow-model-server --grpc_port=666 --model_name=moodel model_base_path=<path to folder with model>

### Running TS in Docker container
    
    docker run -p 666:8500 -p 8501:8501 --mount type=bind,source=<path to folder with model>,target=/models/model -t tensorflow/serving
    
### Script for test model

    python test_classification_server.py
    
### Jupyter notebook for test model

Run Jupyter notebook and run all pipelines
