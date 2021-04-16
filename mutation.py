import random


def add_convblock(part: str):
    conv_1 = str(random.randint(1, 512))
    conv_2 = str(random.randint(1, 512))
    return f'{conv_1}-{conv_2}-{part}'


def add_pool(part: str):
    return f'{round(random.uniform(0., 1.), 1)}-{part}'


def remove_layer(part: str):
    return ''


MUTATION_OPERATIONS = []
MUTATION_OPERATIONS.append(add_convblock)
MUTATION_OPERATIONS.append(add_pool)
MUTATION_OPERATIONS.append(remove_layer)