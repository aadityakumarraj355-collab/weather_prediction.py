marks =int(input("Enter the mark of student: "))

if(marks >= 90):
    grade="A"
elif(marks >= 80 and marks < 90):
    Grade="B"
elif(marks >= 70 and marks < 80):
    Grade="C"
elif(marks >= 60 and marks < 70):
    Grade="D"
else:
     grade="E"
    
print("the grade of the student",Grade)