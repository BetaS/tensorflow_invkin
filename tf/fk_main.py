import matplotlib as mpl
mpl.use('TkAgg')

import os
import numpy as np
from tf.MLPTrainer import MLPTrainer
import matplotlib.pyplot as plt
from GenForKin import generate_tc


def load_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()

        if len(lines) > 0:
            print(len(lines), "test case loaded")
            return lines

    return None


if __name__ == "__main__":
    # create TF networks
    trainer_x = MLPTrainer(input_width=7, output_width=1, hidden_width=30, depth=0, learning_rate=0.5, file_name="checkpoint_rx", from_file=False)
    trainer_y = MLPTrainer(input_width=7, output_width=1, hidden_width=100, depth=0, learning_rate=0.1, file_name="checkpoint_ry1", from_file=False)
    trainer_z = MLPTrainer(input_width=7, output_width=1, hidden_width=100, depth=0, learning_rate=0.5, file_name="checkpoint_rz1", from_file=False)

    # create test set
    test_input = []
    test_output_x = []
    test_output_y = []
    test_output_z = []

    for i in range(1000):
        case = generate_tc()
        test_input.append(case[0:7])
        test_output_x.append(case[10:11])
        test_output_y.append(case[11:12])
        test_output_z.append(case[12:13])

    trainer_x.set_tester(test_input, test_output_x)
    trainer_y.set_tester(test_input, test_output_y)
    trainer_z.set_tester(test_input, test_output_z)

    # Construct model
    print("Optimization Finished!")

    plt.ion()
    fig = plt.figure(figsize=[10, 5])
    ax1 = plt.axes([0.05, 0.05, 0.40, 0.9])
    ax2 = plt.axes([0.55, 0.05, 0.40, 0.9])

    #input("press any key")
    idx = 0
    while True:
        # create training set
        train_input = []
        train_output_x = []
        train_output_y = []
        train_output_z = []
        for i in range(1000):
            case = generate_tc()
            train_input.append(case[0:7])
            train_output_x.append(case[10:11])
            #train_output_y.append(case[11:12])
            #train_output_z.append(case[12:13])

        trainer_x.training(train_input, train_output_x, training_epochs=1, display_epochs=1, batch_size=100)
        #trainer_y.training(train_input, train_output_y, training_epochs=1, display_epochs=1, batch_size=100)
        #trainer_z.training(train_input, train_output_z, training_epochs=1, display_epochs=1, batch_size=100)

        idx += 1
        if idx % 1 == 0:
            ax1.clear()
            ax2.clear()

            ax1.plot(trainer_x.accrs, label="rX valid set error")
            #ax1.plot(trainer_x.avgs, label="rX train set error")
            ax1.plot(trainer_y.accrs, label="rY valid set error")
            #ax1.plot(trainer_y.avgs, label="rY train set error")
            ax1.plot(trainer_z.accrs, label="rZ valid set error")
            #ax1.plot(trainer_z.avgs, label="rZ train set error")
            ax1.set_xlabel('#epochs')
            ax1.set_ylabel('error(deg)')
            ax1.set_title("avg error")
            ax1.legend(loc="upper left")

            ax2.set_xlabel('#epochs')
            ax2.set_ylabel('difference')
            ax2.set_title("pos(deg)")

            for i in range(1):
                ax2.plot(trainer_x.error_set[i],  label="rX")
                ax2.plot(trainer_y.error_set[i],  label="rY")
                ax2.plot(trainer_z.error_set[i],  label="rZ")

            axes = ax2.axis()
            ax2.axhline(y=0, xmin=axes[0], xmax=axes[1], c="blue", linewidth=0.5, zorder=0)
            ax2.legend(loc="upper left")

            fig.canvas.draw()

            idx = 0


    print("Training Finished!")

    trainer.finish()
