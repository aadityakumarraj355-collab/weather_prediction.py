info ={
    "name" : "aadiya kumar",
    # "subjects" : ["python","c","java"],
    # "topics" : ["dict","tuple","list"],
    # "age": 22
    "subject":{
        "phy": 97,
        "chem":98,
        "math":94
    }
}
print(list(info))
print(info["subject"]["chem"])

collection ={1,2,3,4,2,"aaditya"}
print(collection)
print(type(collection))
print(len(collection)) #print len=4