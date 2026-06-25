#average of 3 three number using function
def calc_avg(a,b,c):
    avg =(a+b+c)/3
    return avg

print(calc_avg(4,8,12))

#question1
#wap to print the length of list 
cities =["gurgaon","delhi","bihar","mumbai","chennai"]
heroes =["sushant","prithivi raj singh chauhan","alu arjun"]
def print_len(list):
    print(len(list))
    
print_len(cities)
print_len(heroes)

#question2
#WAF to print elements of list
heroes =["sushant","prithivi raj singh chauhan","alu arjun"]
cities =["gurgaon","delhi","bihar","mumbai","chennai"]
def print_list(list):
    for iteam in list:
        print(iteam,end=", ")
        
print_list(heroes)
print_list(cities)

#WAF to find the factorial of n number
n = 5
def calc_fact(n):
    fact = 1
    for i in range(1,n+1):
        fact *= i
    print(fact)
print("\n")
calc_fact(5)


def convertr(usd_value):
    inr_value = usd_value *83
    print(usd_value,"USD= ",inr_value,"INR")
convertr(100)
    
