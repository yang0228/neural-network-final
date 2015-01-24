from datetime import datetime

class Utility:
    """
    A small utility class for printing things
    """

    def __init__(self):
        self.step = 1

    def get_timestamp(self):
        """
        Gets a nicely formatted timestamp to keep track of output files
        :return:
        """
        raw = datetime.now()
        return str(raw.year).zfill(4) + str(raw.month).zfill(2) + str(raw.day).zfill(2) + "-" + str(raw.hour).zfill(2) + str(raw.minute).zfill(2)

    def print_header(self):
        """
        Prints the standard header for the assignment
        :return:
        """
        print(
              "Project 2 - Artificial Neural Networks & Genetic Algorithms\n"
)
    def print_step(self,message):
        """
        Prints the current step of execution
        :param message:
        :return:
        """
        print(str(self.step) + ". " + message)
        self.step += 1
