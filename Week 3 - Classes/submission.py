# @author: Landon Volkmann

# imports
import numpy as np
import random
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

def print_contents(url: str):
    html = requests.get(url)
    bs_obj = BeautifulSoup(html.content, "html.parser")
    contents = """
    title: {}
    links: {}
    """.format(bs_obj.h1, [link.get("href") for link in bs_obj.find_all("a")])
    print(contents)


def save_links(url: str, output_file_name: str):
    html = requests.get(url)
    bs_obj = BeautifulSoup(html.content, "html.parser")
    out = ""
    for link in bs_obj.find_all("a"):
        if link.get('href') is not None:
            out += str(link.get("href")) + "\n"
    out_file = open(output_file_name, 'w')
    out_file.write(out)


# Question 3: Numpy

def get_random_matrix(row, col, min_lim, max_lim):
    random_numbers = []
    for _ in range(row * col):
        random_numbers.append(random.randrange(min_lim, max_lim + 1))
    matrix = np.array(random_numbers)
    matrix.reshape(row, col)

    return matrix


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
    link = "https://en.wikipedia.org/wiki/Deep_learning"
    print_contents(link)
    save_links(link, "output.txt")

    # Question 3 test

    print(get_random_matrix(3, 5, 0, 20))
