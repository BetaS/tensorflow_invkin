import os
import numpy as np
from tf.MLPTrainer import MLPTrainer
import matplotlib.pyplot as plt



def load_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()

        if len(lines) > 0:
            print(len(lines), "test case loaded")
            return lines

    return None


if __name__ == "__main__":
    # Load them!
    loadpath = "../testset_red.txt"
    datas = load_file(loadpath)

    if not datas:
        print("There is no test case")
        raise FileNotFoundError()

    inputs = []
    outputs = []

    # Parse Data
    for data in datas:
        data = data.split()
        """
        inputs.append(list(map(lambda x: float(x), data[1:])))
        outputs.append(list(map(lambda x: float(x), data[:1])))
        """
        inputs.append(list(map(lambda x: float(x), data[7:])))
        outputs.append(list(map(lambda x: float(x), data[:7])))


    # create TF networks
    trainer = MLPTrainer(inputs, outputs, hidden_width=7, depth=100, learning_rate=0.01)

    # Construct model
    print("Optimization Finished!")

    #input("press any key")
    trainer.training(training_epochs=1000, display_epochs=100, batch_size=100)
    print("Training Finished!")

    trainer.finish()
