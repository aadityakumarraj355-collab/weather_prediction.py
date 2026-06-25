a = int(input("Enter the first Number: "))
b = int(input("Enter the second Number: "))
c = int(input("Enter the third Number: "))
if(a >= b and a >= c):
    print("First Number is Largest")
elif(b >= a and b >= c):
    print("Second Number is Largest")
else:
    print("Third NUmber is Largest")