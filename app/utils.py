import os
from mongoengine import ValidationError

DATAPATH = os.getcwd() + '/data/fitfiles/'


def file_name_validator(file_name: str) -> None:
    if not os.path.exists(file_name):
        raise ValidationError(f'No such file: {file_name}')


def nonnegative_number_validator(number: float) -> None:
    if number <= 0:
        raise ValidationError(f'Must be non-negative: {number}')
