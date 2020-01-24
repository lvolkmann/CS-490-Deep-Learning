# @author Landon Volkmann
# Question 1
"""
    The most notable differences between Python 2 and Python 3 include:
        1. Small syntactical differences (eg print "hello" v print("hello")
        2. Supported libraries
        3. Community support
        4. Python 2 is ASCII where 3 is Unicode
        5. In Python 2, integer division truncates result where Python 3 maintains precision
"""


# Question 2

def reverse_and_remove_two():
    """
    Prompts user for string
    Print string in reverse with the last two characters of original string deleted
    :return:
    """
    user_str = input("Please enter a string: ")
    print(user_str[len(user_str) - 3::-1])


def add_two_numbers() -> int:
    """
    Prompts user for two numbers
    Returns result
    :return:
    """

    return int(input("Please enter first number:")) + int(input("\nPlease enter second number:"))


# Question 3

def pythonify():
    """
    Prompts user for a sentence
    Replace all occurrences of 'python' with 'pythons'
    Print result
    :return:
    """

    user_sentence = input("Please enter a sentence:\n")
    to_replace = "python"
    replacement = "pythons"
    print(user_sentence.replace(to_replace, replacement))


def is_palindrome(some_str: str) -> bool:
    """
    Accepts a string
    Returns whether or not string is palindrome
    :param some_str:
    :return:
    """

    return some_str == some_str[::-1]


def armstrong(some_int: int) -> bool:
    """
    Accepts an int
    Returns whether or not int is an armstrong number
    :param some_int:
    :return:
    """

    string_rep = str(some_int)
    sum_val = 0
    for digit in string_rep:
        sum_val += int(digit) ** 3

    return some_int == sum_val


if __name__ == "__main__":

    reverse_and_remove_two()
    print(add_two_numbers())
    pythonify()

    # Bonus
    print(is_palindrome(input("Input string: ")))
    print(armstrong(int(input("Enter num: "))))
