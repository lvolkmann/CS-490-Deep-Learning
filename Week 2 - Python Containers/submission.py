# @author: Landon Volkmann

# Imports
from typing import List


# Q1

def lbs_to_kg(lst: List[int]) -> List[float]:
    """
    :param lst: list of ints in lbs
    :return: list of floats in kg
    """

    kg_per_lb = 0.453592
    kg_lst = [weight * kg_per_lb for weight in lst]

    return kg_lst


def get_weights_from_user() -> List[int]:
    """
    get number of weights to enter
    get weights input
    :return: list of ints representing weights in lbs
    """

    weights_lst = []

    n = int(input("How many student weights will you enter? "))
    print()
    print("Please enter the weights in pounds.")
    for i in range(n):
        weights_lst.append(int(input("{}: ".format(i + 1))))
        print()

    return weights_lst


def print_weights(lst: List[float]):
    print("kg_lst => ".format(lst), end="")
    for weight in lst:
        print("{:.2f}".format(weight), end=" ")


def run_weights_program():
    lbs_lst = get_weights_from_user()
    kg_lst = lbs_to_kg(lbs_lst)
    print_weights(kg_lst)


# Q2

def string_alternative(some_str: str) -> str:
    """
    :param some_str:
    :return: every other letter of string passed
    """

    return some_str[::2]


# Q3

def find_word_count(file_name: str):
    """
    print word_count in a file for each line
    :param file_name:
    :return: None
    """
    try:
        word_count_dict = {}
        fh = open(file=file_name, mode='r')
        for line in fh:
            for word in line.split(sep=" "):
                word = word.strip()
                word_count_dict[word] = word_count_dict.get(word, 0) + 1
        for word, count in word_count_dict.items():
            print(word, ":", count)
    except FileNotFoundError:
        print("{} could not be found".format(file_name))


if __name__ == "__main__":
    run_weights_program()
    print("\nOUT:", string_alternative(input("IN: ")))
    find_word_count("input.txt")
