import matplotlib as mpl
mpl.use('TkAgg')

import os
import numpy as np
from tf.MLPTrainer import MLPTrainer
import matplotlib.pyplot as plt
from GenForKin import generate_tc

if __name__ == "__main__":
    joint_num = 1
    from_file = False

    # create TF networks
    trainer = MLPTrainer(input_width=3, output_width=1, hidden_width=30, depth=0, learning_rate=0.01, file_name="checkpoint_ik_q%d" % (joint_num + 1), from_file=from_file)

    # create test set
    test_input = []
    test_output = []

    for i in range(1000):
        case = generate_tc()
        test_input.append(case[7:10])
        test_output.append(case[joint_num:joint_num+1])

    trainer.set_tester(test_input, test_output)

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
        train_output = []
        for i in range(1000):
            case = generate_tc()
            train_input.append(case[7:10])
            train_output.append(case[joint_num:joint_num+1])

        trainer.training(train_input, train_output, training_epochs=1, display_epochs=1, batch_size=100)

        idx += 1
        if idx % 1 == 0:
            ax1.clear()
            ax2.clear()

            ax1.plot(trainer.accrs, label="valid set error")
            ax1.plot(trainer.avgs, label="train set error")
            ax1.set_xlabel('#epochs')
            ax1.set_ylabel('error(mm3)')
            ax1.set_title("avg error")
            ax1.legend(loc="upper left")

            ax2.set_xlabel('#epochs')
            ax2.set_ylabel('difference')
            ax2.set_title("joint angle(deg)")

            ax2.plot(trainer.error_set[0],  label="q%d"%(joint_num+1))

            axes = ax2.axis()
            ax2.axhline(y=0, xmin=axes[0], xmax=axes[1], c="blue", linewidth=0.5, zorder=0)
            ax2.legend(loc="upper left")

            fig.canvas.draw()

            idx = 0


    print("Training Finished!")

    trainer.finish()
