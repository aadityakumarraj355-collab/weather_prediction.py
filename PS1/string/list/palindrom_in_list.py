list1 =[1,2,3,3,2,1]
list2 = [1,2,3]

copy_list1 = list1.copy()
copy_list1.reverse()

if(copy_list1 == list1):
    print("list1 is palindrom")
else:
    print("list1 is not a palindrom")
    
copy_list2 = list2.copy()
copy_list2.reverse()

if(copy_list2 == list2):
    print("list2 is palindrom")
else:
    print("list2 is not a palindrom")
    