# @author: Landon Volkmann

# imports
import numpy as np
import gc
from bs4 import BeautifulSoup
import requests


# Question 1: Employee Class

class Employee(object):

    EMPLOYEE_COUNT = 0

    def __init__(self, name: str, family: str, department: str):
        self.name = name
        self.family = family
        self.department = department

        # Increment class attributes
        Employee.increment_employee_count()

    @staticmethod
    def get_employee_count():
        return Employee.EMPLOYEE_COUNT

    @classmethod
    def increment_employee_count(cls):
        cls.EMPLOYEE_COUNT += 1


class FullTimeEmployee(Employee):

    def __init__(self, name: str, family: str, department: str, salary: float):
        super().__init__(name, family, department)
        self.salary = salary

    @property
    def salary(self):
        return self._salary

    @salary.setter
    def salary(self, value):
        self._salary = value

    @staticmethod
    def get_average_salary():

        total_salary = 0
        ft_emp_count = 0

        for inst in gc.get_objects():
            if isinstance(inst, FullTimeEmployee):
                total_salary += inst.salary
                ft_emp_count += 1

        return total_salary / ft_emp_count


# Question 2: Web Scraping

def print_title(url: str):
    """Prints title of given url"""
    html = requests.get(url)
    bs_obj = BeautifulSoup(html.content, "html.parser")
    contents = """
    {}
    """.format(bs_obj.title)
    print(contents)


def save_links(url: str, output_file_name: str):
    """Saves all links of given url to outbound file"""
    html = requests.get(url)
    bs_obj = BeautifulSoup(html.content, "html.parser")
    out = ""
    for link in bs_obj.find_all("a"):
        if link.get('href') is not None:
            out += str(link.get("href")) + "\n"
    out_file = open(output_file_name, 'w')
    out_file.write(out)


# Question 3: Numpy

def random_then_replace(row: int, col: int, min_lim: int, max_lim: int):
    """
    Generates random array of specified dimensions and values
    Replaces each row maximum with 0
    """

    matrix = np.random.randint(min_lim, max_lim, row*col)
    matrix = matrix.reshape(row, col)

    print("Original:")
    print(matrix)

    print()
    print("Replaced")

    # get max by row and convert from (n, ) -> (n, 1)
    row_maxes = matrix.max(axis=1).reshape(-1, 1)
    matrix[:] = np.where(matrix == row_maxes, 0, matrix)

    print(matrix)


if __name__ == "__main__":

    # Question 1 test
    emp1 = Employee("Landon", "none", "IT")
    emp2 = Employee("Trevor", "some", "IT")

    print(Employee.get_employee_count())

    ft_emp_1 = FullTimeEmployee("Dougy", "alot", "synergy", 70000.00)
    ft_emp_2 = FullTimeEmployee("Bobby", "too much", "kitchen", 30000.00)

    print(Employee.get_employee_count())
    print(FullTimeEmployee.get_average_salary())

    # Question 2 test
    url_name = "https://en.wikipedia.org/wiki/Deep_learning"
    print_title(url_name)
    save_links(url_name, "output.txt")

    # Question 3 test
    random_then_replace(3, 5, 0, 20)
