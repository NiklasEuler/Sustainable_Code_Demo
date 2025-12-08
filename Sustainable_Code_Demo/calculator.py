def addition(a, b):
    """Performs addition of two numbers."""
    return a + b

def subtraction(a, b):
    """Performs subtraction of two numbers."""
    return a - b

def multiplication(a, b):
    """Performs multiplication of two numbers."""
    return a * b

def division(a, b):
    """Performs division of a by b with error handling for division by zero."""
    if b == 0:
        raise ZeroDivisionError("Division by zero is not allowed.")
    return a / b