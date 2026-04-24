# Numbers
num = 1
int = 2.5
print(num, int)

# List
my_list = [1,2,3]
result = my_list[len(my_list)-1]
print(result)

# Dict
my_dict = {
    "name":"Aman",
    "designation":"Software Dev"
}

name = my_dict["name"]
designation = my_dict.get("designation")

print(name,designation)

# Tuple
mixed_tuple = ("Aman",26,"software developer")
print(mixed_tuple[len(mixed_tuple)-1])