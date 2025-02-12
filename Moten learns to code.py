
choice = '1'

def add(x, y):
    return x + y


try:
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))
except ValueError:
    print("Invalid input. Please enter a number.")

if choice == '1':
    print(num1, "+", num2, "=", add(num1, num2))