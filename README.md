# Self driving car
Note! This is just a research project and it should not be used for a real self driving car.

This project is a upgraded version of original sully chen self driving model.This is implemented in tensorflow. Some minor changes have been made to nvidia's end to end deeplearning architecture rest is same. A block diagram of our training system is shown in Figure below.  Images are fed into a CNN which then computes a proposed steering command.  The proposed command is compared to the desired command for that image and the weights of the CNN are adjusted to bring the CNN output closer to the desired output. The weight adjustment is accomplished using back propagation as implemented in the tensorflow.
